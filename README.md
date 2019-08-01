# Demo link

http://3.18.91.191/

# Paper link

Nie, Allen, Erin D. Bennett, and Noah D. Goodman. "Learning to Explain: Answering Why-Questions via Rephrasing." arXiv preprint arXiv:1906.01243 (2019). (https://arxiv.org/abs/1906.01243)



# Download Learning to Explain Corpus

If you wish to have a chatbot that learns to answer why-questions in a chitchat style, you can train on our corpus! 

Here is the link to download them:

```bash
aws s3 cp --recursive --no-sign-request --region=us-west-1 s3://learning2explain/ .
```

We present two datasets: `because` and `because_ctx`. The later one includes 5 previous sentences as context. This command does not require login or authentication.

You can view a list of items from this link: https://s3-us-west-2.amazonaws.com/learning2explain/


# Web Server Setup

![Demo Image](https://github.com/windweller/L2EWeb/blob/master/L2EDemoImage.png?raw=true)

This code depends on an older version of OpenNMT. The version is zipped and uploaded as part
of this code base.

You can unzip and install that version of OpenNMT through:

```bash
cd OpenNMT-py
pip install -e .
```

Also, we need to set up Stanford CoreNLP server as well and have it running in order to 
parse the sentences. The way to start the StanfordNLP server process can be followed in https://github.com/erindb/corenlp-ec2-startup.

```
bash SERVE.sh
```

Then you want to download the L2E `.pt` model file from the AWS as well and create a folder and save the file to it:

```bash
mkdir model
mv ~/learning2explain/models/L2E-final-model/dissent_step_80000.pt model/
```

Start the server by calling:

```bash
sudo python3 main.py
```

# Integrate L2E into your system

We have written a simple wrapper around OpenNMT-py (our zipped version). Note that we also sens HTTP call to the Stanford CoreNLP dependency parser implicitly.

```python
from seq2seq import PhenomenonEncoder, L2EDecoder
encoder = PhenomenonEncoder(model_file_path, temp_dir, logger)
decoder = L2EDecoder(encoder)
```

More details on how we build the pipeline are in `api.py`.
