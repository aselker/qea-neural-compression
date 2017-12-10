### Neural Compression

This is an algorithm designed to compress text using a neural network and Huffman codings.  It is written in pure Python, using Numpy.  It was created for QEA at Olin College in 2017.  Currently, the algorithm does not work; consider it an educational tool above all.  

The algorithm compresses text in three steps, and decompresses it in two.  The compressed data consists of a series of Huffman-coded letters and the weights for a neural network; the details of how these are created and then used will be covered in the next few sections.  

### The Corpus

This algorithm is designed to work with a large corpus, such as a book.  Inlcuded in the repository is a copy of _Moby-Dick_, by Herman Melville.  'clean\_file.py' strips out all characters except for letters and spaces, replacing them with underscores.  It also converts all letters to lowercase.  This is because the neural network trains and runs faster with fewer letters.  In the future, common punctuation such as periods and line breaks should be included in the network, and special cases should be added to deal with rarer characters, such as the Hebrew letters near the beginning.  

In order for the compression to be successful, there must be patterns in the text which the network can pick up on.  Normal English text fits this requirement, as would code, markup, or even some types of binary data.  This algorithm cannot compress data that has already been compressed, encrypted, or otherwise obfuscated.  

### The Neural Network

The neural network is a recurrent network using the hyperbolic tangent as its activation function, arranged into fully-connected layers, each of which has some recurrent neurons.  With the current configuration, there are six layers, each of which has 28 feedforward neurons and 24 recurrent neurons.  Input to the network is fed into the 28 input neurons; all are set to an activation of -0.5 except for the one which corresponds to the current letter, which is set to 0.5.  The letter 'c', for instance, is represented by  
`-0.5, -0.5, 0.5, -0.5, ... -0.5`  
Space and underscore are the last two entries in the list, respectively.  When the network renders output, it does so in a similar form, assigning each possible letter a likelihood.  The network is trained on the corpus it will eventually compress, given each letter (and the recurrent state) as input and asked to predict the next one.  

### Huffman Codings

This algorithm is based around the interaction between a predictive neural net and a series of Huffman codings.  After the network is trained, it is given the same inputs it was trained on (the corpus), and asked at each step to predict the next letter, rendering a list of likelihoods.  Using these likelihoods, a Huffman coding tree is created for the letter, and the actual letter is encoded using that tree.  The coding is then saved, and the network moves on to the next step.  This list of codings, along with the weights of the trained network, make up the compressed file.

### Decompression

To recover the compressed corpus, a neural network is constructed that is identical to the one used to compress the corpus.  This is possible because the connection weights are stored in the file.  The network is asked to predict the first letter of the document.  Once again, a Huffman tree is created for the letter using the predicted probabilities, and the actual first letter is found by decoding the first stored letter using this tree.  The network is then asked to predict the second letter, and so on.  

Because an accurate prediction means that the Huffman-coded letter will take less space to store, creating a smaller file relies on the network being effective at predicting letters.  This does not, however, affect the accuracy of the decompressed file, which is always lossless.  As long as each prediction at compression time can be repeated at decompression time, the Huffman trees will be identical, and so will the files.

### Performance

The algorithm does not work.  Small errors accumulate in the evaluation of the network in the decompression phase, leading to a garbled output.  This is probably because of inherent imprecision in floating-point arithmetic, which should not be relied upon to have repeatable results.

The algorithm's performance, in terms of compression ratio and speed, has not yet been measured.  Right now the compression ratio is probably abysmal compared to well-established algorithms like DEFLATE, and may in fact be worse than no compression at all, after including the stored connection weights. Performance is closely tied to the neural network's success at predicting the text, which is currently low.  Replacing the network half of the backend with a more performant, better-tuned piece of software such as Pytorch could greatly improve both compression ratio and speed.
