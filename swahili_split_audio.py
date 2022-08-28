import pandas as pd
import split_test_2, os
import sklearn
from catboost import CatBoostClassifier

data_filename = './data/ALFFA_dataset_ allosaurus vs epitran.csv'
test_name = "swahili_cross"

def main(iterations):
    #first load all the data in
    data = pd.read_csv( data_filename )

    #The data for training is from the text data.
    data_for_training = list(data['cleaned_transcript_epitran'])
    data_for_training = [s if isinstance(s,str) else "" for s in data_for_training]

    #The data for adding spaces is from audio data.
    data_for_spacing = list(data['allosaurus_transcript_no_spaces'])
    data_for_spacing = [s if isinstance(s,str) else "" for s in data_for_spacing]

    model_name = f"{test_name}_model_{iterations}.cb"


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

    
    #Run the model and add a column on the dataset and save it back out.
    respaced_result = []
    edit_distances_to_with_spaces = []
    reductions_per_space = []
    reductions = []
    for i,audio_line in enumerate(data_for_spacing):
        spaces_added = split_test_2.add_whitespace( model, audio_line, quite=True )
        respaced_result.append( spaces_added )

        #see if the spaces helped the edit distance.
        edit_distance_before = split_test_2.levenshteinDistance( data_for_training[i], audio_line )
        edit_distance_after = split_test_2.levenshteinDistance( data_for_training[i], spaces_added )
        reduction = edit_distance_before-edit_distance_after
        edit_distances_to_with_spaces.append( edit_distance_after )

        num_spaces_added = len( spaces_added ) - len( audio_line )
        reduction_per_space = reduction/num_spaces_added if num_spaces_added > 0 else 0
        reductions_per_space.append(reduction_per_space)
        reductions.append(reduction)

        print(  f"{i} of {len(data_for_spacing)}: edit reduction {reduction}, per space {reduction_per_space},  num_added {num_spaces_added} {spaces_added}\n" )

    data['allosaurus_transcript_spaced'] = respaced_result
    data['allosaurus_transcript_spaced_edit_distance'] = edit_distances_to_with_spaces
    data['allosaurus_transcript_edit_reduction' ] = reductions
    data['allosaurus_transcript_edit_reduce_per_space' ] = reductions_per_space
    data.to_csv( f"{test_name}_{iterations}.csv" )

    print( f"finished {test_name} with {iterations} iterations" )

if __name__ == "__main__":
    main( 4500 )
    main( 6500 )