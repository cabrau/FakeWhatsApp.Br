# FakeWhatsApp.Br
An annotated Corpus of anonymized WhatsApp messages in PT-BR public groups for automatic detection of textual misinformation and low credibility users. To get detailed information about the construction and experimentation of the corpus, check out our paper published in ICEIS 2021 conference:

If you use our corpus, please include a citation to our corresponding paper.

## Data
The data collected during 2018 brazilian presidential ellections is located at:
* ``data/2018/fakeWhatsApp.BR_2018.csv``

The data is stored in a CSV file, where each line is a message sent in a public group. The dictionary of variables is the following:
* ``id``: unique ID of a user
* ``date``: day of the year that the message was sent
* ``ddi``: international identifier
* ``country``: country assigned to the ddi
* ``country_iso3``: ISO3 code of country
* ``ddd``: regional brazilian telephone code
* ``state``: brazilian state
* ``midia``: boolean variable indicating if the message is a media file (1) or not (0)
* ``url``: boolean variable indicating if the message contains an url (1) or don't (0)
* ``characters``: number of characters in message's text
* ``words``: number of words in message's text
* ``viral``: boolean variable indicating if a message with the exactly same text and more of 5 words appears in the corpus (1) or don't (0). The viral messages were the ones manually labelled.
* ``shares``: number of times that a message with the exactly same text appears in the corpus
* ``text``: textual content of message
* ``misinformation``: manually assigned label if the message contains misinformation (1) or don't (1). The value -1 means that the message was not labelled.
* ``ambiguity``: (not implemented yet) assigned 1 if the misinformation label is uncertain: the veracity of the text cannot be confirmed or it can be confused with a personal opinion, humorous text, etc.

## Notebooks:
* ``1 - parser.ipynb``<br>
This notebook parses the data collected in WhatsApp groups, converting from free text format to structured data in a CSV table.

* ``2 - labeling and anonymization.ipynb``<br>
In this notebook we transfer the labels annotated manually in the viral messages to the entire corpus and remove personal data such as phone numbers present in the text.

* ``3 - exploratory analysis.ipynb``<br>
Exploration and visualization of the data set.

* ``4 - compare corpora.ipynb``<br>
Comparison with fake news corpus on Twitter to demonstrate the need for a corpus of WhatsApp texts.

* ``5 - misinformation detection ml.ipynb``<br>
Experiments with classical machine learning models to classify textual misinformation.

* ``6 - deep learning char level cnn.ipynb``<br>
Experiments with a character level convolutional neural network to classify textual misinformation.

* ``7 - user features.ipynb``<br>
Exploiting user features to detect misinformation

* ``8 - user classification.ipynb``<br>
Experiments classifying users as superspreaders

* ``9 - automatic dataset expansion.ipynb``<br>
Experiments with automatic expansion of dataset using cosine similarity

* ``10 - user credibility.ipynb``<br>
Modeling user credibility to improve misinformation detection