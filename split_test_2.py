
import pandas as pd
import sklearn
import sklearn.model_selection
from catboost import CatBoostClassifier, Pool
import os

bible_filename = "/home/lansford/Sync/projects/tf_over/golden_path/joey_test/temp/target_compile/pre_tokenizer_dump.txt"
num_pre_context_chars = 7
num_post_context_chars = 7

def load_file( filename ):
    result = []
    with open(filename, "rt") as fin:
        for line in fin:
            result.append(line)
    return result



#edit distance from https://stackoverflow.com/a/32558749/1419054
def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def strip_whitespace( content ):
    return content.replace( " ", "" ).replace( "\n", "" )


def parse_for_training( lines ):

    lines = "\n".join(lines)

    contexts = []
    answers = []

    for index in range(len(lines)):
        #pre_context has spaces in it
        pre_context = lines[max(index-num_pre_context_chars,0):index]
        pre_context = ((" " * num_pre_context_chars) + pre_context)[-num_pre_context_chars:]

        #post_context doesn't.
        double_post_length_with_whitespace = lines[index:min(index+(num_post_context_chars*2),len(lines))]
        post_length_without_whitespace = strip_whitespace(double_post_length_with_whitespace) + (" " * num_post_context_chars)
        post_context = post_length_without_whitespace[:num_post_context_chars]

        full_context = pre_context + post_context

        contexts.append( full_context )

        #now add a true if the current index is a return or space.
        answers.append( 1 if lines[index] in [' ', '\n'] else 0 )

    context_split_into_dict = {}

    for i in range( num_pre_context_chars+num_post_context_chars ):
        this_slice = []
        for context in contexts:
            this_slice.append( context[i] )
        context_split_into_dict['c' + str(i) ] = this_slice


        
    return pd.DataFrame( context_split_into_dict ), pd.DataFrame( {'predict_space':answers} )

def train_catboost( X_train,X_test, y_train, y_test, iterations, learning_rate = .07 ):
    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=list(range(X_train.shape[1]))
    )
    validation_pool = Pool(
        data=X_test,
        label=y_test,
        cat_features=list(range(X_train.shape[1]))
    )
    model = CatBoostClassifier(
        iterations = iterations,
        learning_rate = learning_rate
    )
    model.fit( train_pool, eval_set=validation_pool, verbose=True )

    print( 'Model is fitted: {}',format(model.is_fitted()))
    print('Model params:\n{}'.format(model.get_params()))

    return model

def add_whitespace( model, text, quite=False ):
    result = ""
    for i in range(len(text)):
        pre_context = ( (" " * num_pre_context_chars) + result[max(0,len(result)-num_pre_context_chars):])[-num_pre_context_chars:]
        post_context = (text[i:min(len(text),i+num_post_context_chars)] + (" " * num_post_context_chars))[:num_post_context_chars]
        full_context = pre_context + post_context
        context_as_dictionary = { 'c'+str(c):[full_context[c]] for c in range(len(full_context)) }
        context_as_pd = pd.DataFrame( context_as_dictionary )

        model_result = model.predict( context_as_pd )[0]

        if not quite and len( result ) % 500 == 0: print( "%" + str(i*100/len(text))[:4] + " " + result[-100:])

        if model_result: result += " "
        result += text[i]

        pass
    return result


def main( iterations = 50 ):
    #first load all the bible in
    full_bible = load_file( bible_filename )

    #split it between training and testing
    lines_for_training = int(.5*len(full_bible) )
    bible_for_training = full_bible[:lines_for_training]
    bible_for_testing = full_bible[lines_for_training:]

    #see if the model exists yet.
    if not os.path.exists( f"split_test2_model_{iterations}.cb" ):

        #parse the training side into structure which can train a model
        X,y = parse_for_training( bible_for_training )

        #split it between train and test.
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X,
            y,
            random_state=1,
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        #train catboost model
        model = train_catboost( X_train,X_test, y_train, y_test, iterations )

        #save the model
        model.save_model( f"split_test2_model_{iterations}.cb" )
    else:
        model = CatBoostClassifier().load_model(  f"split_test2_model_{iterations}.cb" )

    
    #test model on remaining bible.
    bible_for_testing_without_whitespace = strip_whitespace(" ".join( bible_for_testing ))
    white_space_added = add_whitespace( model, bible_for_testing_without_whitespace )

    #save it out.
    with open( f"split_output2_{iterations}.txt", "wt" ) as fout:
        fout.write( white_space_added )
    
def gradio_demo(iterations):
    import gradio as gr 

    #first load all the bible in
    full_bible = load_file( bible_filename )
    #just train on all of it because the test is playing with it
    bible_for_training = full_bible

    model_filename = f"split_test2_model_full_{iterations}.cb"

        #see if the model exists yet.
    if not os.path.exists( model_filename ):

        #parse the training side into structure which can train a model
        X,y = parse_for_training( bible_for_training )

        #split it between train and test.
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X,
            y,
            random_state=1,
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        #train catboost model
        model = train_catboost( X_train,X_test, y_train, y_test, iterations )

        #save the model
        model.save_model( model_filename )
    else:
        model = CatBoostClassifier().load_model(  model_filename )

    
    def gradio_function( text ):
        text = text.replace( " ", "" )
        return add_whitespace( model, text, quite=True )

    with gr.Blocks() as demo:
        inp = gr.Textbox( label="Input" )
        out = gr.Textbox( label="Output" )
        inp.change( gradio_function, inputs=[inp], outputs=[out] )
    demo.launch( share=True )

if __name__ == '__main__': 
    #main(200)
    # for i in range(10):
    #     main((i+1)*10)
    gradio_demo(200)