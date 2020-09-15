# FakeWhatsApp.Br
An annotated Corpus of WhatsApp messages for automatic detection of textual misinformation.

## Notebooks:
* 1 - parser.ipynb<br/>
This notebook parses the data collected in WhatsApp groups, converting from free text format to structured data in a CSV table.

* 2 - labeling and anonymization.ipynb<br/>
In this notebook we transfer the labels annotated manually in the viral messages to the entire corpus and remove personal data such as phone numbers present in the text.

* 3 - exploratory analysis.ipynb<br/>
Exploration and visualization of the data set.

* 4 - compare corpora<br/>
Comparison with fake news corpus on Twitter to demonstrate the need for a corpus of WhatsApp texts.

* 5 - misinformation detection ml.ipynb
Experiments with classical machine learning models to classify textual misinformation.

* 6 - deep learning char level cnn.ipynb
Experiments with a character level convolutional neural network to classify textual misinformation.