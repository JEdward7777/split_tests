import pandas as pd
import split_test_2
import os
import sklearn
from catboost import CatBoostClassifier

data_filename = './data/ALFFA_dataset_ allosaurus vs epitran.csv'
test_name = "swahili_test"


def main(iterations, learning_rate):

    #print( data.describe() )


    #first load all the data in
    data = pd.read_csv( data_filename )
    full_data = list(data['cleaned_transcript_epitran'])
    full_data = [s if isinstance(s,str) else "" for s in full_data]

    #split it between training and testing
    lines_for_training = int(.5*len(full_data) )
    data_for_training = full_data[:lines_for_training]
    data_for_testing = full_data[lines_for_training:]

    model_name = f"{test_name}_model_{iterations}_{learning_rate}.cb"

    #see if the model exists yet.
    if not os.path.exists( model_name ):

        #parse the training side into structure which can train a model
        X,y = split_test_2.parse_for_training( data_for_training )

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
        model = split_test_2.train_catboost( X_train,X_test, y_train, y_test, iterations )

        #save the model 
        model.save_model( model_name )

        #200 test of .0434672
        #500 test of .0379901
    else:
        model = CatBoostClassifier().load_model(  model_name )

    
    #Test model on remaining data and save it out to csv file for results.
    edit_distance_sum = 0
    with open( f"{test_name}_{iterations}_{learning_rate}.csv", "wt" ) as fout:
        fout.write( "truth,output,edit distance\n" )

        for i,testing_line in enumerate(data_for_testing):
            without_whitespace = split_test_2.strip_whitespace( testing_line )
            spaces_added = split_test_2.add_whitespace( model, without_whitespace, quite=True )
            edit_distance = split_test_2.levenshteinDistance( testing_line, spaces_added )
            result = f"{testing_line},{spaces_added},{edit_distance}"
            fout.write( f"{result}\n" )
            print(  f"{i} of {len(data_for_testing)}: {result}\n" )

            edit_distance_sum += edit_distance

    print( f"Finished {iterations} iterations with a learning_rate of {learning_rate} and got average edit_distance of {edit_distance_sum/len(data_for_testing)}")


    # #test model on remaining data.
    # data_for_testing_without_whitespace = split_test_2.strip_whitespace(" ".join( data_for_testing ))
    # white_space_added = split_test_2.add_whitespace( model, data_for_testing_without_whitespace )

    # #save it out.
    # with open( f"{test_name}_{iterations}.txt", "wt" ) as fout:
    #     fout.write( white_space_added )
    

if __name__ == '__main__': 
    #main(1000,.07)
    #main(1000,.04)
    #main(1000,.1)
    #main(1000,.15)
    #main(1000,.2)
    main(8000,.07)