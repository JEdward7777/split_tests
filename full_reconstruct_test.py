import pandas as pd
import sklearn, os, re
import sklearn.model_selection
from catboost import CatBoostClassifier, Pool, CatBoostError



MATCH = 0
DELETE_FROM = 1
INSERT_TO = 2
START = 3

class edit_trace_hop():
    parrent = None
    edit_distance = None
    char = None
    from_row_i = None
    to_column_i = None
    action = None

    def __str__( self ):
        if self.action == START:
            return "<start>"
        elif self.action == INSERT_TO:
            return f"<ins> {self.char}"
        elif self.action == DELETE_FROM:
            return f"<del> {self.char}"
        elif self.action == MATCH:
            return f"<match> {self.char}"
        return "eh?"

    def __repr__( self ):
        return self.__str__()

def trace_edits( from_sentance, to_sentance, print_debug=False ):
    #iterating from will be the rows down the left side.
    #iterating to will be the columns across the top.
    #we will keep one row as we work on the next.

    last_row = None
    current_row = []

    #the index handles one before the index in the string
    #to handle the root cases across the top and down the left of the
    #match matrix.
    for from_row_i in range( len(from_sentance)+1 ):

        for to_column_i in range( len(to_sentance )+1 ):

            best_option = None

            #root case.
            if from_row_i == 0 and to_column_i == 0:
                best_option = edit_trace_hop()
                best_option.parrent = None
                best_option.edit_distance = 0
                best_option.char = ""
                best_option.from_row_i = from_row_i
                best_option.to_column_i = to_column_i
                best_option.action = START

            #check left
            if to_column_i > 0:
                if best_option is None or current_row[to_column_i-1].edit_distance + 1 < best_option.edit_distance:
                    best_option = edit_trace_hop()
                    best_option.parrent = current_row[to_column_i-1]
                    best_option.edit_distance = best_option.parrent.edit_distance + 1
                    best_option.char = to_sentance[to_column_i-1]
                    best_option.from_row_i = from_row_i
                    best_option.to_column_i = to_column_i
                    best_option.action = INSERT_TO
            
            #check up
            if from_row_i > 0:
                if best_option is None or last_row[to_column_i].edit_distance + 1 < best_option.edit_distance:
                    best_option = edit_trace_hop()
                    best_option.parrent = last_row[to_column_i]
                    best_option.edit_distance = best_option.parrent.edit_distance + 1
                    best_option.char = from_sentance[from_row_i-1]
                    best_option.from_row_i = from_row_i
                    best_option.to_column_i = to_column_i
                    best_option.action = DELETE_FROM

                #check match
                if to_column_i > 0:
                    if to_sentance[to_column_i-1] == from_sentance[from_row_i-1]:
                        if best_option is None or last_row[to_column_i-1].edit_distance <= best_option.edit_distance: #prefer match so use <= than <
                            best_option = edit_trace_hop()
                            best_option.parrent = last_row[to_column_i-1]
                            best_option.edit_distance = best_option.parrent.edit_distance + 1
                            best_option.char = from_sentance[from_row_i-1]
                            best_option.from_row_i = from_row_i
                            best_option.to_column_i = to_column_i
                            best_option.action = MATCH

            if best_option is None: raise Exception( "Shouldn't end up with best_option being None" )
            current_row.append(best_option)

        last_row = current_row
        current_row = []

    if print_debug:
        def print_diffs( current_node ):
            if current_node.parrent is not None:
                print_diffs( current_node.parrent )
            
            if current_node.action == START:
                print( "start" )
            elif current_node.action == MATCH:
                print( f"match {current_node.char}" )
            elif current_node.action == INSERT_TO:
                print( f"insert {current_node.char}" )
            elif current_node.action == DELETE_FROM:
                print( f"del {current_node.char}" )
        print_diffs( last_row[-1] )
    return last_row[-1]

def list_trace( trace ):
    if trace.parrent is None:
        result = [trace]
    else:
        result = list_trace( trace.parrent )
        result.append( trace )
    return result

def parse_single_for_training( from_sentance, to_sentance, num_pre_context_chars=7, num_post_context_chars=7 ):
    trace = trace_edits( from_sentance, to_sentance )

    #we will collect a snapshot at each step.
    trace_list = list_trace(trace)


    training_collection = []

    #execute these things on the from_sentance and see if we get the to_sentance.
    working_from = from_sentance
    working_to = ""
    used_from = ""
    continuous_added = 0
    continuous_dropped = 0
    for thing in trace_list:
        #gather action and context for training
        if thing.action != START:
            from_context = (working_from + (" " * num_post_context_chars))[:num_post_context_chars]
            to_context =   ((" " * num_pre_context_chars) + working_to )[-num_pre_context_chars:]
            used_context = ((" " * num_pre_context_chars) + used_from  )[-num_pre_context_chars:]

            training_collection.append({
                "from_context": from_context,
                "to_context": to_context,
                "used_context": used_context,
                "action": thing.action,
                "continuous_added": continuous_added,
                "continuous_dropped": continuous_dropped,
                "char": thing.char if thing.action == INSERT_TO else ' ',
            })

        #now execute the action for the next step.
        if thing.action == START:
            pass
        elif thing.action == INSERT_TO:
            working_to += thing.char
            continuous_added += 1
            continuous_dropped = 0
        elif thing.action == DELETE_FROM:
            used_from += working_from[0]
            working_from = working_from[1:]
            continuous_added = 0
            continuous_dropped += 1
        elif thing.action == MATCH:
            used_from += working_from[0]
            working_to += working_from[0]
            working_from = working_from[1:]
            continuous_added = 0
            continuous_dropped = 0

    
    if to_sentance != working_to:
        print( "Replay failure" )

    #so now I have training_collection which is a list of dictionaries where each dictionary is an action with a context.
    #I need to change it into a dictionary of lists where each dictionary a column and the lists are the rows.
    context_split_into_dict = {}

    #first collect the from_context:
    for i in range( num_post_context_chars ):
        this_slice = []
        for training in training_collection:
            this_slice.append( training['from_context'][i] )
        context_split_into_dict[ f"f{i}" ] = this_slice
    
    #now collect to_context:
    for i in range( num_pre_context_chars ):
        this_slice = []
        for training in training_collection:
            this_slice.append( training['to_context'][i] )
        context_split_into_dict[ f"t{i}" ] = this_slice

    #now collect used_context
    for i in range( num_pre_context_chars ):
        this_slice = []
        for training in training_collection:
            this_slice.append( training['used_context'][i] )
        context_split_into_dict[ f"u{i}" ] = this_slice

    
    #now these two things.
    context_split_into_dict["continuous_added"] = []
    context_split_into_dict["continuous_dropped"] = []
    for training in training_collection:
        context_split_into_dict["continuous_added"].append( training["continuous_added"] )
        context_split_into_dict["continuous_dropped"].append( training["continuous_dropped"] )

    #now also collect the output answers.
    result_split_into_dict = {}
    action_slice = []
    char_slice = []
    for training in training_collection:
        action_slice.append( training['action'] )
        char_slice.append( training['char'] )
    result_split_into_dict['action'] = action_slice
    result_split_into_dict['char']   = char_slice
        
    #now return it as a dataframe.
    return pd.DataFrame( context_split_into_dict ), pd.DataFrame( result_split_into_dict )
        
def parse_for_training( from_sentances, to_sentances ):
    out_observations_list = []
    out_results_list = []

    for index, (from_sentance, to_sentance) in enumerate(zip( from_sentances, to_sentances )):
        if type(from_sentance) != float and type(to_sentance) != float: #bad lines are nan which are floats.
            specific_observation, specific_result = parse_single_for_training( from_sentance, to_sentance )

            out_observations_list.append( specific_observation )
            out_results_list.append( specific_result )
        if index % 100 == 0:
            print( f"parsing {index} of {len(from_sentances)}")

    return pd.concat( out_observations_list ), pd.concat( out_results_list )

def train_catboost( X_train,X_test, y_train, y_test, X, y, iterations, learning_rate = .07, use_gpu=False ):
    #as we aren't using test, we might as well train on that split data as well.
    X_train = X
    y_train = y

    X_test_limited = X_test
    y_test_limited = y_test
    X_train = X_train.fillna( ' ' )
    passed = False
    while not passed:
        #try:
            train_pool = Pool(
                data=X_train,
                label=y_train,
                cat_features=list(range(X_train.shape[1]))
            )
            # validation_pool = Pool(
            #     data=X_test_limited,
            #     label=y_test_limited,
            #     cat_features=list(range(X_test.shape[1]))
            # )
            validation_pool = None #hack to get around chars being found that aren't allowed.
            model = CatBoostClassifier(
                iterations = iterations,
                learning_rate = learning_rate,
                task_type="GPU" if use_gpu else "CPU",
                devices='0:1' if use_gpu else None
            )
            model.fit( train_pool, eval_set=validation_pool, verbose=True )
            passed = True
        # except CatBoostError as cbe:
        #     #examples in the test set have chars not see in the training set so remove them.
        #     error_string = str(cbe)
        #     test_index_matcher = re.compile( r'test #(\d+) contains' )
        #     match = test_index_matcher.search( error_string )
        #     bad_test_index = int(match[1])
        #     X_test_limited = X_test_limited.drop(bad_test_index).reset_index(drop=True)
        #     y_test_limited = y_test_limited.drop(bad_test_index).reset_index(drop=True)

    print( 'Model is fitted: {}',format(model.is_fitted()))
    print('Model params:\n{}'.format(model.get_params()))

    return model

def split_and_train( X, y, iterations, use_gpu ):
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
    print( "training" )
    model = train_catboost( X_train,X_test, y_train, y_test, X, y, iterations, use_gpu=use_gpu )
    return model

def train_reconstruct_models( from_sentances, to_sentances, parse_cache_name_in, parse_cache_name_out, iterations, use_parse_caches, use_gpu ):
    if not os.path.exists( parse_cache_name_in ) or not use_parse_caches:
        X,Y = parse_for_training( from_sentances, to_sentances )
        X.to_csv( parse_cache_name_in,   index=False )
        Y.to_csv( parse_cache_name_out,  index=False )
    else:
        X = pd.read_csv( parse_cache_name_in  )
        Y = pd.read_csv( parse_cache_name_out )

    #train and save the action_model
    action_model = split_and_train( X, Y['action'], iterations, use_gpu=use_gpu )

    #and the char model
    #slice through where only the action is insert.
    insert_indexes = Y['action'] == INSERT_TO
    char_model = split_and_train( X[insert_indexes], Y['char'][insert_indexes], iterations, use_gpu=use_gpu )

    return action_model, char_model

def do_reconstruct( action_model, char_model, text, quite=False, num_pre_context_chars=7, num_post_context_chars=7 ):
    # result = ""
    # for i in range(len(text)):
    #     pre_context = ( (" " * num_pre_context_chars) + result[max(0,len(result)-num_pre_context_chars):])[-num_pre_context_chars:]
    #     post_context = (text[i:min(len(text),i+num_post_context_chars)] + (" " * num_post_context_chars))[:num_post_context_chars]
    #     full_context = pre_context + post_context
    #     context_as_dictionary = { 'c'+str(c):[full_context[c]] for c in range(len(full_context)) }
    #     context_as_pd = pd.DataFrame( context_as_dictionary )

    #     model_result = model.predict( context_as_pd )[0]

    #     if not quite and len( result ) % 500 == 0: print( "%" + str(i*100/len(text))[:4] + " " + result[-100:])

    #     if model_result: result += " "
    #     result += text[i]

    #     pass
    # return result

    #test for nan.
    if text != text: text = ''

    working_from = text
    working_to = ""
    used_from = ""
    continuous_added = 0
    continuous_dropped = 0
    while working_from and len(working_to) < 3*len(text) and (len(working_to) < 5 or working_to[-5:] != (working_to[-1] * 5)):
        from_context = (working_from + (" " * num_post_context_chars))[:num_post_context_chars]
        to_context =   ((" " * num_pre_context_chars) + working_to )[-num_pre_context_chars:]
        used_context = ((" " * num_pre_context_chars) + used_from  )[-num_pre_context_chars:]

        #construct the context.
        context_as_dictionary = {}
        #from_context
        for i in range( num_post_context_chars ):
            context_as_dictionary[ f"f{i}" ] = [from_context[i]]
        #to_context
        for i in range( num_pre_context_chars ):
            context_as_dictionary[ f"t{i}" ] = [to_context[i]]
        #used_context
        for i in range( num_pre_context_chars ):
            context_as_dictionary[ f"u{i}" ] = [used_context[i]]
        #these two things.
        context_as_dictionary["continuous_added"]   = [continuous_added]
        context_as_dictionary["continuous_dropped"] = [continuous_dropped]

        #make it a pandas.
        context_as_pd = pd.DataFrame( context_as_dictionary )

        #run the model
        action_model_result = action_model.predict( context_as_pd )[0][0]

        if action_model_result == START:
            pass
        elif action_model_result == INSERT_TO:
            #for an insert ask the char model what to insert
            char_model_result = char_model.predict( context_as_pd )[0][0]

            working_to += char_model_result
            continuous_added += 1
            continuous_dropped = 0
        elif action_model_result == DELETE_FROM:
            used_from += working_from[0]
            working_from = working_from[1:]
            continuous_added = 0
            continuous_dropped += 1
        elif action_model_result == MATCH:
            used_from += working_from[0]
            working_to += working_from[0]
            working_from = working_from[1:]
            continuous_added = 0
            continuous_dropped = 0

    return working_to

#edit distance from https://stackoverflow.com/a/32558749/1419054
def levenshteinDistance(s1, s2):
    if s1 != s1: s1 = ''
    if s2 != s2: s2 = ''
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


def main(iterations=6000, use_gpu=False, flush_cache=False, from_column = 'cleaned_transcript_epitran', to_column = 'cleaned_transcript' ):
    data_filename = './data/ALFFA_dataset_ allosaurus vs epitran.csv'
    
    print( "loading csv" )
    full_data = pd.read_csv( data_filename )

    action_model_name = f"full_reconstruct_test_{iterations}__action.cb"
    char_model_name = f"full_reconstruct_test_{iterations}__char.cb"

    parse_cache_name_in = f"full_reconstruct_train_cache_in.csv"
    parse_cache_name_out = f"full_reconstruct_train_cache_out.csv"


    #use half the data for training.
    half_index = int(.5*len(full_data))
    train_data = full_data.iloc[:half_index,:].reset_index(drop=True)
    test_data = full_data.iloc[half_index:,:].reset_index(drop=True)

    #see if the model exists yet.
    if flush_cache or not os.path.exists( action_model_name ) or not os.path.exists( char_model_name ):

        #parse the training side into structure which can train a model
        print( "parcing data for training" )

        action_model, char_model = train_reconstruct_models( from_sentances=train_data[from_column], 
                to_sentances=train_data[to_column], 
                parse_cache_name_in = parse_cache_name_in,
                parse_cache_name_out = parse_cache_name_out, 
                iterations = iterations,
                use_gpu = use_gpu,
                use_parse_caches = not flush_cache )

        action_model.save_model( action_model_name )
        char_model.save_model( char_model_name )
    else:
        action_model = CatBoostClassifier().load_model(  action_model_name )
        char_model   = CatBoostClassifier().load_model(  char_model_name   )



    #process the test data, compute the resulting edit distances
    #and save it back out.
    test_result = []
    before_edit_distances = []
    after_edit_distances = []
    improvement = []
    for row in range(len( test_data )):
        test_result.append(
            do_reconstruct( action_model, char_model, test_data[from_column][row] )
        )
    
        before_edit_distances.append(
            levenshteinDistance( test_data[from_column][row], test_data[to_column][row] )
        )

        after_edit_distances.append(
            levenshteinDistance( test_result[row], test_data[to_column][row] )
        )

        improvement.append(
            before_edit_distances[-1] - after_edit_distances[-1]
        )
    pd_results = pd.DataFrame( {
        "in_data": test_data[from_column],
        "out_data": test_data[to_column],
        "generated_data": test_result,
        "before_edit_distance": before_edit_distances,
        "after_edit_distance": after_edit_distances,
        "improvement": improvement,
    })
    pd_results.to_csv( f"full_reconstruct_results_{iterations}_b.csv" )

def gradio_demo(iterations):
    import gradio as gr 

    #just train on all of it because the test is playing with it

    action_model_name = f"full_reconstruct_test_{iterations}__action.cb"
    char_model_name = f"full_reconstruct_test_{iterations}__char.cb"


    action_model = CatBoostClassifier().load_model(  action_model_name )
    char_model   = CatBoostClassifier().load_model(  char_model_name   )
    
    def gradio_function( text ):
        return do_reconstruct( action_model, char_model, text )

    with gr.Blocks() as demo:
        inp = gr.Textbox( label="Input" )
        out = gr.Textbox( label="Output" )
        inp.change( gradio_function, inputs=[inp], outputs=[out] )
    demo.launch( share=False )

if __name__ == '__main__':
    # main(100, False )
    # main(200, False )
    # main(1000, False )
    # main(2000, False )
    # main(4000, False )
    # main(8000, False )
    #gradio_demo(8000)

    #main( 100, flush_cache=True, from_column='allosaurus_transcript_no_spaces', to_column='cleaned_transcript_epitran')
    #main( 300, flush_cache=True, from_column='allosaurus_transcript_no_spaces', to_column='cleaned_transcript_epitran')
    #main( 1000, flush_cache=True, from_column='allosaurus_transcript_no_spaces', to_column='cleaned_transcript_epitran')
    #main( 2001, use_gpu=True, flush_cache=True, from_column='allosaurus_transcript_no_spaces', to_column='cleaned_transcript_epitran')
    #main( 2002, use_gpu=True, flush_cache=True, from_column='allosaurus_transcript_no_spaces', to_column='cleaned_transcript_epitran')
    main( 102, flush_cache=True, from_column='allosaurus_transcript_no_spaces', to_column='cleaned_transcript_epitran')

#trace_edits( "matokeo ja ut͡ʃaɠuzi mkuu wa nt͡ʃi ja cote ɗe ivoiɾe inajoonɡoza kwa uzaliʃaʄi wa kakao ɗuniani", "matokeo ya uchaguzi mkuu wa nchi ya cote de ivoire inayoongoza kwa uzalishaji wa kakao duniani", print_debug=True )
#trace_edits( "", "", print_debug=True )

# results = parse_single_for_training( "matokeo ja ut͡ʃaɠuzi mkuu wa nt͡ʃi ja cote ɗe ivoiɾe inajoonɡoza kwa uzaliʃaʄi wa kakao ɗuniani", "matokeo ya uchaguzi mkuu wa nchi ya cote de ivoire inayoongoza kwa uzalishaji wa kakao duniani" )

# print("Result is" + str(results) )