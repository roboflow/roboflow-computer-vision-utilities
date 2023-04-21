from roboflow import Roboflow


# obtaining your API key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
rf = Roboflow(api_key="INSERT_PRIVATE_API_KEY")
workspace = rf.workspace()

raw_data_location = "INSERT_PATH_TO_IMAGES"
raw_data_extension = ".jpg"

# replace 1 with your model version number for inference
inference_endpoint = ["INSERT_MODEL_ID", 1]
upload_destination = "INSERT_MODEL_ID"

# set the conditionals values as necessary for your active learning needs
# NOTE - not all conditional fields are required
conditionals = {
    "required_objects_count" : 1,
    "required_class_count": 1,
    "target_classes": [],
    "minimum_size_requirement" : float('-inf'),
    "maximum_size_requirement" : float('inf'),
    "confidence_interval" : [10,90],
}

## filtering out images for upload by similarity is available for paid plans
## contact the Roboflow team for access: https://roboflow.com/sales
# conditionals = {
#     "required_objects_count" : 1,
#     "required_class_count": 1,
#     "target_classes": [],
#     "minimum_size_requirement" : float('-inf'),
#     "maximum_size_requirement" : float('inf'),
#     "confidence_interval" : [10,90],
#     "similarity_confidence_threshold": .3,
#     "similarity_timeout_limit": 3
# }

workspace.active_learning(raw_data_location=raw_data_location, 
    raw_data_extension=raw_data_extension,
    inference_endpoint=inference_endpoint,
    upload_destination=upload_destination,
    conditionals=conditionals)
