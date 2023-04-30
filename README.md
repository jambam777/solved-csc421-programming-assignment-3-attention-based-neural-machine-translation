Download Link: https://assignmentchef.com/product/solved-csc421-programming-assignment-3-attention-based-neural-machine-translation
<br>
<strong>Submission: </strong>You must submit two files through MarkUs<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>: a PDF file containing your writeup, titled a3-writeup.pdf, and your code file nmt.ipynb. Your writeup must be typeset.

The programming assignments are individual work. See the Course Information handout<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> for detailed policies.

You should attempt all questions for this assignment. Most of them can be answered at least partially even if you were unable to finish earlier questions. If you were unable to run the experiments, please discuss what outcomes you might hypothetically expect from the experiments. If you think your computational results are incorrect, please say so; that may help you get partial credit.

<h1>Introduction</h1>

In this assignment, you will train a few attention-based neural machine translation models to translate words from English to Pig-Latin. Along the way, you’ll gain experience with several important concepts in NMT, including <em>gated recurrent neural networks </em>and <em>attention</em>.

<h2>Pig Latin</h2>

Pig Latin is a simple transformation of English based on the following rules (applied on a per-word basis):

<ol>

 <li>If the first letter of a word is a <em>consonant</em>, then the letter is moved to the end of the word, and the letters “ay” are added to the end: team→eamtay.</li>

 <li>If the first letter is a <em>vowel</em>, then the word is left unchanged and the letters “way” are added to the end: impress→impressway.</li>

 <li>In addition, some consonant pairs, such as “sh”, are treated as a block and are moved to the end of the string together: shopping→oppingshay.</li>

</ol>

To translate a whole sentence from English to Pig-Latin, we simply apply these rules to each word independently:

i went shopping→iway entway oppingshay

We would like a neural machine translation model to learn the rules of Pig-Latin <em>implicitly</em>, from (English, Pig-Latin) word pairs. Since the translation to Pig Latin involves moving characters around in a string, we will use <em>character-level </em>recurrent neural networks for our model.

Because English and Pig-Latin are so similar in structure, the translation task is almost a copy task; the model must remember each character in the input, and recall the characters in a specific order to produce the output. This makes it an ideal task for understanding the capacity of NMT models.

<h1>Setting Up</h1>

We recommend that you use <a href="https://colab.research.google.com/"><strong>Colab</strong></a><a href="https://colab.research.google.com/">(</a><a href="https://colab.research.google.com/">https://colab.research.google.com/</a><a href="https://colab.research.google.com/">)</a> for the assignment, as all the assignment notebooks have been tested on Colab. Otherwise, if you are working on your own environment, you will need to install Python 2, PyTorch (<a href="https://pytorch.org/">https://pytorch.org</a><a href="https://pytorch.org/">)</a>, iPython Notebooks, SciPy, NumPy and scikit-learn. Check out the websites of the course and relevant packages for more details.

From the assignment zip file, you will find one python notebook file: nmt.ipynb. To setup the Colab environment, you will need to upload the two notebook files using the upload tab at <a href="https://colab.research.google.com/">https://colab.research.google.com/</a><a href="https://colab.research.google.com/">.</a>

<h2>Data</h2>

The data for this task consists of pairs of words where the <em>source s</em><sup>(<em>i</em>) </sup>is an English word, and the <em>target t</em><sup>(<em>i</em>) </sup>is its translation in Pig-Latin. The dataset is composed of unique words from the book “Sense and Sensibility,” by Jane Austen. The vocabulary consists of 29 tokens: the 26 standard alphabet letters (all lowercase), the dash symbol -, and two special tokens &lt;SOS&gt; and &lt;EOS&gt; that denote the start and end of a sequence, respectively. <a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> The dataset contains 6387 unique (English, Pig-Latin) pairs in total; the first few examples are:

{ (the, ethay), (family, amilyfay), (of, ofway), … }

In order to simplify the processing of <em>mini-batches </em>of words, the word pairs are grouped based on the lengths of the source and target. Thus, in each mini-batch the source words are all the same length, and the target words are all the same length. This simplifies the code, as we don’t have to worry about batches of variable-length sequences.

<h1>Part 1: Encoder-Decoder Models and Teacher-Forcing [2 mark]</h1>

Translation is a <em>sequence-to-sequence </em>problem: in our case, both the input and output are sequences of characters. A common architecture used for seq-to-seq problems is the encoder-decoder model [2], composed of two RNNs, as follows:

The encoder RNN compresses the input sequence into a fixed-length vector, represented by the final hidden state <em>h<sub>T</sub></em>. The decoder RNN conditions on this vector to produce the translation, character by character.

Input characters are passed through an embedding layer before they are fed into the encoder RNN; in our model, we learn a 29 × 10 embedding matrix, where each of the 29 characters in the vocabulary is assigned a 10-dimensional embedding. At each time step, the decoder RNN outputs a vector of <em>unnormalized log probabilities </em>given by a linear transformation of the decoder hidden state. When these probabilities are normalized, they define a distribution over the vocabulary, indicating the most probable characters for that time step. The model is trained via a cross-entropy loss between the decoder distribution and ground-truth at each time step.

Encoder                                                                   Decoder

Figure 1: Training the NMT encoder-decoder architecture.

Encoder                                                                   Decoder

Figure 2: Generating text with the NMT encoder-decoder architecture.

The decoder produces a distribution over the output vocabulary conditioned on the previous hidden state and the output token in the previous timestep. A common practice used to train NMT models is to feed in the <em>ground-truth token </em>from the previous time step to condition the decoder output in the current step. This training procedure is known as “teacher-forcing” shown in Figure 1. At test time, we don’t have access to the ground-truth output sequence, so the decoder must condition its output on the token it generated in the previous time step, as shown in Figure 2.

<strong>Conceptual Questions</strong>

<ol>

 <li>How do you think the architecture in Figure 1 will perform on long sequences, and why? Consider the amount of information the decoder gets to see about the input sequence.</li>

 <li>What are some techniques / modifications we can use to improve the performance of this architecture on long sequences? List at least two.</li>

 <li>What problem may arise when training with teacher forcing? Consider the differences that arise when we switch from training to testing.</li>

 <li>Can you think of any way to address this issue? Read the abstract and introduction of the paper “<em>Scheduled sampling for sequence prediction with recurrent neural networks</em>” [1], and answer this question in your own words.</li>

</ol>

<h1>Part 3: Gated Recurrent Unit (GRU) [2 marks]</h1>

Throughout the rest of the assignment, you will implement some attention-based neural machine translation models, and finally train the model and examine the results.

Open the notebook nmt.ipynb on Colab and answer the following questions.

<ol>

 <li>The forward pass of a Gated Recurrent Unit is defined by the following equations:</li>

</ol>

<em>r</em><em>t </em>= <em>σ</em>(<em>W</em><em>irx</em><em>t </em>+ <em>W</em><em>hrh</em><em>t</em>−1 + <em>b</em><em>r</em>)        (1) <em>z</em><em>t </em>= <em>σ</em>(<em>W</em><em>izx</em><em>t </em>+ <em>W</em><em>hzh</em><em>t</em>−1 + <em>b</em><em>z</em>)  (2)

))                                                  (3)

(4)

where  is the element-wise multiplication. Although PyTorch has a GRU built in (nn.GRUCell), we’ll implement our own GRU cell from scratch, to better understand how it works. The notebook has been divided into different sections. Find the GRU cell section of the notebook. Complete the __init__ and forward methods of the MyGRUCell class, to implement the above equations. A template has been provided for the forward method.

<ol start="2">

 <li>Train the GRU RNN in the “Training – RNN decoder” section. (Make sure you run all the previous cells to load the training and utility functions.)</li>

</ol>

By default, the script runs for 100 epochs. At the end of each epoch, the script prints training and validation losses, and the Pig-Latin translation of a fixed sentence, “the air conditioning is working”, so that you can see how the model improves qualitatively over time. The script also saves several items to the directory h20-bs64-rnn:

<ul>

 <li>The best encoder and decoder model paramters, based on the validation loss.</li>

 <li>A plot of the training and validation losses.</li>

</ul>

How do the results look, qualitatively? Does the model do better for certain types of words than others?

<ol start="3">

 <li>Use this model to translate words in the next notebook cell using translate_sentence Try a few of your own words by changing the variable TEST_SENTENCE. Which failure modes can you identify?</li>

</ol>

<h1>Part 4: Implementing Attention [4 marks]</h1>

Attention allows a model to look back over the input sequence, and focus on relevant input tokens when producing the corresponding output tokens. For our simple task, attention can help the model remember tokens from the input, e.g., focusing on the input letter c to produce the output letter c.

The hidden states produced by the encoder while reading the input sequence,can be viewed as <em>annotations </em>of the input; each encoder hidden state <em>h<sup>enc</sup><sub>i </sub></em>captures information about the <em>i<sup>th </sup></em>input token, along with some contextual information. At each time step, an attention-based decoder computes a <em>weighting </em>over the annotations, where the weight given to each one indicates its relevance in determining the current output token.

In particular, at time step <em>t</em>, the decoder computes an attention weight  for each of the encoder hidden states <em>h<sup>enc</sup><sub>i </sub></em>. The attention weights are defined such that 0 1 and

<ol>

 <li>is a function of an encoder hidden state and the previous decoder hidden state,), where <em>i </em>ranges over the length of the input sequence.</li>

</ol>

There are a few engineering choices for the possible function <em>f</em>. In this assignment, we will implement two different attention models: 1) the additive attention using a two-layer MLP and 2) the scaled dot product attention, which measures the similarity between the two hidden states.

To unify the interface across different attention modules, we consider attention as a function whose inputs are triple (queries, keys, values), denoted as (<em>Q,K,V </em>).

<ol>

 <li>In the additive attention, we will <em>learn </em>the function <em>f</em>, parameterized as a two-layer fullyconnected network with a ReLU activation. This network produces unnormalized weights that are used to compute the final context vector:</li>

</ol>

<em>,</em>

(<em>t</em>)                                            <sup>(<em>t</em>)</sup>

<em>α<sub>i        </sub></em>= softmax(˜<em>α     </em>)<em><sub>i</sub>,</em>

<em>T c</em><em>t </em>= X<em>α</em><em>i</em>(<em>t</em>)<em>V</em><em>i.</em>

<em>i</em>=1

Here, the notation [<em>Q<sub>t</sub></em>;<em>K<sub>i</sub></em>] denotes the concatenation of vectors <em>Q<sub>t </sub></em>and <em>K<sub>i</sub></em>. To obtain the attention weights in between 0 and 1, we apply the softmax function over the unnormalized attention. Once we have the attention weights, a <em>context vector c<sub>t </sub></em>is computed as a linear combination of the encoder hidden states, with coefficients given by the weights.

<strong>Implement the additive attention mechanism.               </strong>Fill in the forward methods of the

AdditiveAttention class. Use the self.softmax function in the forward pass of the AdditiveAttention class to normalize the weights.

<table width="552">

 <tbody>

  <tr>

   <td width="166">hidden_sizeDecoder Hidden States</td>

   <td width="249">hidden_sizeEncoder Hidden States</td>

   <td width="138">1Attention Weights</td>

  </tr>

 </tbody>

</table>

batch_size                      batch_size

batch_size      seq_len                        seq_len

Figure 3: Dimensions of the inputs, Decoder Hidden States (<em>query</em>), Encoder Hidden States (<em>keys/values</em>) and the attention weights (<em>α</em><sup>(<em>t</em>)</sup>).

For the forward pass, you are given a batch of query of the current time step, which has dimension batch_size x hidden_size, and a batch of keys and values for each time step of the input sequence, both have dimension batch_size x seq_len x hidden_size. The goal is to obtain the context vector. We first compute the function <em>f</em>(<em>Q<sub>t</sub>,K</em>) for each query in the batch and <em>all </em>corresponding keys <em>K<sub>i</sub></em>, where <em>i </em>ranges over seq_len different values. You must do this in a vectorized fashion. Since <em>f</em>(<em>Q<sub>t</sub>,K<sub>i</sub></em>) is a scalar, the resulting tensor of attention weights should have dimension batch_size x seq_len x 1. Some of the important tensor dimensions in the AdditiveAttention module are visualized in Figure 3. The AdditiveAttention module should return both the context vector batch_size x 1 x hidden_size and the attention weights batch_size x seq_len x 1.

Depending on your implementation, you will need one or more of these functions (click to jump to the PyTorch documentation):

<ul>

 <li><a href="http://pytorch.org/docs/0.3.0/torch.html#torch.squeeze">squeeze</a></li>

 <li><a href="http://pytorch.org/docs/0.3.0/torch.html#torch.unsqueeze">unsqueeze</a></li>

 <li><a href="http://pytorch.org/docs/0.3.0/tensors.html?highlight=expand_as#torch.Tensor.expand_as">expand</a> <a href="http://pytorch.org/docs/0.3.0/tensors.html?highlight=expand_as#torch.Tensor.expand_as">as</a></li>

 <li><a href="http://pytorch.org/docs/0.3.0/torch.html?highlight=cat#torch.cat">cat</a></li>

 <li><a href="http://pytorch.org/docs/0.3.0/tensors.html?highlight=view#torch.Tensor.view">view</a></li>

 <li><a href="http://pytorch.org/docs/0.3.0/tensors.html?highlight=bmm#torch.bmm">bmm</a></li>

</ul>

We have provided a template for the forward method of the AdditiveAttention class. You are free to use the template, or code it from scratch, as long as the output is correct.

<ol start="2">

 <li>We will now apply the AdditiveAttention module to the RNN decoder. You are given a batch of decoder hidden states as the query, , for time <em>t </em>− 1, which has dimension batch_size x hidden_size, and a batch of encoder hidden states as the keys and values, <em>annotations</em>), for each timestep in the input sequence, which has</li>

</ol>

dimension batch_size x seq_len x hidden_size.

We will use these as the inputs to the self.attention to obtain the context. The output context vector is concatenated with the input vector and passed into the decoder GRU cell at each time step, as shown in Figure 4.

Figure 4: Computing a context vector with attention.

<strong>Fill in </strong>the forward method of the RNNAttentionDecoder class, to implement the interface shown in Figure 4. You will need to:

(a) Compute the context vector and the attention weights using self.attention (b) Concatenate the context vector with the current decoder input.

(c) Feed the concatenation to the decoder GRU cell to obtain the new hidden state.

<ol start="3">

 <li>Train the Attention RNN in the “Training – RNN attention decoder” section. How do the results compare to RNN decoder without attention for certain type of words? Can you identity any failure mode? How does the training speed compare? Why?</li>

 <li>In lecture, we learnt about Scaled Dot-product Attention used in the transformer models. The function <em>f </em>is a dot product between the linearly transformed query and keys using weight matrices <em>W<sub>q </sub></em>and <em>W<sub>k</sub></em>:</li>

</ol>

<em>,</em>

= softmax(˜<em>α</em><sup>(<em>t</em>)</sup>)<em><sub>i</sub>,</em>

<em>T c</em><em>t </em>= X<em>α</em><em>i</em>(<em>t</em>)<em>W</em><em>vV</em><em>i,</em>

<em>i</em>=1

where, <em>d </em>is the dimension of the query and the <em>W<sub>v </sub></em>denotes weight matrix project the value to produce the final context vectors.

<strong>Implement the scaled dot-product attention mechanism. </strong>Fill in the __init__ and forward methods of the ScaledDotAttention class. Use the PyTorch <a href="https://pytorch.org/docs/stable/torch.html#torch.bmm">torch.bmm</a> to compute the dot product between the batched queries and the batched keys in the forward pass of the ScaledDotAttention class for the unnormalized attention weights. Your forward pass <strong>needs to </strong>work with both 2D query tensor (batch_size x (1) x hidden_size) <strong>and </strong>3D query tensor (batch_size x k x hidden_size).

Because we use the same interface between different attention modules, we can reuse the previous RNN attention decoder with the scaled dot-product attention.

Train the Attention RNN using scaled dot-product attention in the “Training – RNN scaled dot-product attention decoder” section. How do the results and training speed compare to the additive attention? Why is there such different?

<h1>Part 5: Attention is All You Need [2 mark]</h1>

<ol>

 <li>What are the advantages and disadvantages of using additive attention vs scaled dot-product attention? List one advantage and one disadvantage for each method.</li>

 <li>Fill in the forward method in the CausalScaledDotAttention. It will be mostly the same as the ScaledDotAttention The additional computation is to mask out the attention to the future time steps. You will need to add self.neg_inf to some of the entries in the unnormalized attention weights. You may find <a href="https://pytorch.org/docs/stable/torch.html#torch.tril">torch.tril</a> handy for this part.</li>

 <li>We will now use ScaledDotAttention as the building blocks for a simplified transformer[3] decoder. You are given a batch of decoder input embeddings, <em>x<sup>dec </sup></em>across all time steps, which has dimension batch_size x decoder_seq_len x hidden_size. and a batch of encoder hidden states,<em>annotations</em>), for each time step in the input sequence, which has dimension batch_size x encoder_seq_len x hidden_size.</li>

</ol>

The transformer solves the translation problem using layers of attention modules. In each layer, we first apply the CausalScaledDotAttention self-attention to the decoder inputs followed by ScaledDotAttention attention module to the encoder annotations, similar to the attention decoder from the previous question. The output of the attention layers are fed into an hidden layer using ReLU activation. The final output of the last transformer layer are passed to the self.out to compute the word prediction. To improve the optimization, we add residual connections between the attention layers and ReLU layers. The simple transformer architecture is shown in Figure 5

Figure 5: Computing the output of a transformer layer.

Fill in the forward method of the TransformerDecoder class, to implement the interface shown in Figure 5.

Train the transformer in the “Training – Transformer decoder” section. How do the translation results compare to the previous decoders? How does the training speed compare?

<ol start="4">

 <li>Modify the transformer decoder __init__ to use non-causal attention for both self attention and encoder attention. What do you observe when training this modified transformer? How do the results compare with the causal model? Why?</li>

 <li>In the lecture, we mentioned the transformer encoder will be able to learn the ordering of its inputs without the explicit positional encoding. Why does our simple transformer decoder work without the positional encoding?</li>

</ol>

<h1>Part 6: Attention Visualizations [2 marks]</h1>

One of the benefits of using attention is that it allows us to gain insight into the inner workings of the model. By visualizing the attention weights generated for the input tokens in each decoder step, we can see where the model focuses while producing each output token. In this part of the assignment, you will visualize the attention learned by your model, and try to find interesting success and failure modes that illustrate its behaviour.

The Attention visualization section loads the model you trained from the previous section and uses it to translate a given set of words: it prints the translations and display heatmaps to show how attention is used at each step. endcenter

<ol>

 <li>Visualize different attention models using your own word by modifying TEST_WORD_ATTN. Since the model operates at the character-level, the input doesn’t even have to be a real word in the dictionary. You can be creative! You should examine the generated attention maps. Try to find failure cases, and hypothesize about why they occur. Some interesting classes of words you may want to try are:</li>

</ol>

<ul>

 <li>Words that begin with a single consonant (e.g., <em>cake</em>).</li>

 <li>Words that begin with two or more consonants (e.g., <em>drink</em>).</li>

 <li>Words that have unusual/rare letter combinations (e.g., <em>aardvark</em>).</li>

 <li>Compound words consisting of two words separated by a dash (e.g., <em>well-mannered</em>). These are the hardest class of words present in the training data, because they are long, and because the rules of Pig-Latin dictate that each part of the word (e.g., <em>well </em>and <em>mannered</em>) must be translated separately, and stuck back together with a dash: <em>ellway-annerdmay</em>.</li>

 <li>Made-up words or toy examples to show a particular behaviour.</li>

</ul>

<strong>Include attention maps for both success and failure cases in your writeup, along with your hypothesis about why the models succeeds or fails.</strong>

<h1>What you need to submit</h1>

<ul>

 <li>One code file: ipynb.</li>

 <li>A PDF document titled a3-writeup.pdf containing your answers to the conceptual questions, and the attention visualizations, with explanations.</li>

</ul>

<h1>References</h1>

<ul>

 <li>Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. In <em>Advances in Neural Information Processing Systems</em>, pages 1171–1179, 2015.</li>

 <li>Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In <em>Advances in neural information processing systems</em>, pages 3104–3112, 2014.</li>

 <li>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, L ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In <em>Advances in Neural Information Processing Systems</em>, pages 5998–6008, 2017.</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> <a href="https://markus.teach.cs.toronto.edu/csc421-2019-01">https://markus.teach.cs.toronto.edu/csc421-2019-01</a>

<a href="#_ftnref2" name="_ftn2">[2]</a> <a href="http://cs.toronto.edu/~rgrosse/courses/csc421_2019/syllabus.pdf">http://cs.toronto.edu/</a><a href="http://cs.toronto.edu/~rgrosse/courses/csc421_2019/syllabus.pdf">~</a><a href="http://cs.toronto.edu/~rgrosse/courses/csc421_2019/syllabus.pdf">rgrosse/courses/csc421_2019/syllabus.pdf</a>

<a href="#_ftnref3" name="_ftn3">[3]</a> Note that for the English-to-Pig-Latin task, the input and output sequences share the same vocabulary; this is not always the case for other translation tasks (i.e., between languages that use different alphabets).