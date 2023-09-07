# svm_textgen

Repository to host examples that attempt to do text generation, by providing a text file with custom data. Much more than text generation,
with the tests done up until this time, this will serve as a tool to do Question Answering over the provided data.

The biggest of the influences of this script is the following paper:
- https://arxiv.org/abs/2308.16898

This first examples have been AI generated mainly with changes here and there. The potential of improvement is huge, somehow it feels hard
to find a way to train the SVMs, in order to approximate the behavior of SOTA LLMs nowadays.

# Examples
## Generating Metasploit commands:
As an example, a python script to train a SVM with a short list of metasploit commands is given; the possibilities or creating a synergy between
generative AI and cybersecurity are huge, and would bring a whole new world of possibilities, to find better ways to architecture information,
suggestions from AI models can ease and speed up the process to find the right tools, commands and parameters during an audit or the resolution
of a cybersecurity issue.
