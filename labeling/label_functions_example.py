from label_functions import *

# This script demonstrates how functions in annotations_functions can be used.
# most of these functions are standalone and do not require that you've run them in order. 
# all functions expect working directory to be endometriosis, with labels stored in:
#    - endometriosis/labeling/prodigy
# and annotation output stored in:
#    - endometriosis/labeling/prodigy/output



# if you have csvs (from multi-category labels) already formatted (with format_raw_annotations) and 
# you want to combine them with new, model-in-the-loop single label annotations,
# the combine_multiple_jsons function can do this:

annotation_file_names = ['family_annotations-0.jsonl', 'family_annotations-1.jsonl']
existing_csv_names = ['relationships.csv']
output_file_name = 'combined_family.csv'
key_col = 'FAMILY'
input_path = os.getcwd()

final_df = combine_multiple_jsons(input_path, annotation_file_names, existing_csv_names, output_file_name, key_col)


# determine the type of label
label_type = "relations"

# get file paths --> useful for double checking, but most functions already embed this function
output_path, labels_file_path, annotations_path, output_csv_path = get_file_paths_for_label(label_type)

# only need to run this after a new prodigy output --> reformats from json to csv, with labels properly formatted
format_raw_annotations(label_type)

# simple printout of label stats
get_label_stats(label_type)

# load annotations (what is created by format_raw_annotations())
annotations_df = load_annotations(label_type)

# get the classification accuracy as more data is added --> will plot over increased nums of samples
classification_accuracy_by_data_amount(label_type)

# get the overall cross validated accuracy for each label
accuracy_scores = get_accuracy_metrics(label_type)
