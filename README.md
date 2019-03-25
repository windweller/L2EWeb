# Setup

This code depends on an older version of OpenNMT. The version is zipped and uploaded as part
of this code base.

You can unzip and install that version of OpenNMT through:

```bash
pip install -e .
```

Also, we need to set up Stanford CoreNLP server as well and have it running in order to 
parse the sentences.

Start the server by:

```bash
sudo python3 main.py
```