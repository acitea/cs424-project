First of all for the segmentation task, there are several things to take note:

- "custom-dataset" folder:
    - Refers to a popular manga dataset that was also used to train and test Magi.
    - It uses a popmanga_test.py Dataset builder script with some specified schema
    The script requires an annotations.zip file which contains:

    A file at annotations/mangaplus_links.txt with URLs to manga chapters
    Split files at annotations/seen.txt and annotations/unseen.txt
    JSON annotation files for each image


    The script uses mloader to download images from the links in mangaplus_links.txt
    The dataset will have two splits: "seen" and "unseen"
    Each example in the dataset contains:

    image_path: Path to the manga image
    magi_annotations: Bounding box annotations with labels
    character_clusters: Character cluster IDs
    text_char_matches: Text to character matches
    text_tail_matches: Text tail matches
    text_classification: Text classification labels
    character_names: Names of characters

    I run this dataloader from popmanga_test.ipynb


- "magiV2" folder
    - Creates proper mapping for each character cluster to a character, then synthesizes this info to pass into the model's predict method
    - The dataset contains labels [0,1,2] which correspond to ["character", "text", "panel"] respectively
    - I've created a panel segmentation and output that into "panel_outputs" but you will notice all the manga are not really from any one single manga series. It would take way too much time to annotate all the One Piece manga chapters.
    - The model than creates several text to character mappings
