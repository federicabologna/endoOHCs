# TO RUN THIS SCRIPT make sure to use a Python environment set up for Prodigy!

# prodigy CLASS_TYPE DB_NAME DATA_PATH --label "LABEL"

# LABELS:
#       relationships.txt:
#           FAMILY
#           FRIEND
#           PARTNER
#           THERAPIST
#           DOCTORS
#           ENDO SUPPORT COMMUNITY
#
#       topics.txt:
#           ENDOMETRIOSIS SYMPTOMS
#           DISBELIEF
#
#       intent.txt:
#           SEEKING INFORMATIONAL SUPPORT
#           PROVIDING INFORMATIONAL SUPPORT
#           SEEKING EMOTIONAL SUPPORT
#           PROVIDING EMOTIONAL SUPPORT

######################################################

# INTENT LABELS
# prodigy textcat.manual endo-posts-intent file_path_to_label_data --label intent.txt

# INTENT LABELS
# prodigy textcat.manual endo-posts-topics file_path_to_label_data --label topics.txt

# INTENT LABELS
# prodigy textcat.manual endo-posts-relations file_path_to_label_data --label relationships.txt



