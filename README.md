
TextSumma
=
Just give it a shot for reproducing the ACL 2016 paper [*Neural Summarization by Extracting Sentences and Words*](https://arxiv.org/abs/1603.07252).

## Quick Start
- **Step1 : Obtain datasets**  
   Go [*here*](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) to download the corpus and get the scripts of ***one-billion-word-language-modeling-benchmark*** for training the word vectors. Run this and see more [*details*](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark/blob/master/README.corpus_generation):
   ```bash
   $ tar --extract -v --file ../statmt.org/tar_archives/training-monolingual.tgz --wildcards training-monolingual/news.20??.en.shuffled
   $ ./scripts/get-data.sh
   ```
   The dataset ***cnn-dailymail*** with highlights in this paper offered by the authors is in [*here*](https://docs.google.com/uc?id=0B0Obe9L1qtsnSXZEd0JCenIyejg&export=download).  
- **Step2 : Preprocess**  
  Run this script to training the word vectors in the dataset ***one-billion-word-language-modeling-benchmark***:  
  ```bash
  $ python train_w2v.py './one-billion-word-benchmark' 'output_gensim_model' 'output_word_vector'
  ```
  Run this script to extract the sentences, labels and entitys in the dataset ***cnn-dailymail*** and get them pickled:
  ```bash
  $ python
  ``` 
- **Step3 : Install nvidia-docker**  
  Go for GPUs acceleration. See [*installation*](https://github.com/NVIDIA/nvidia-docker) to get more information for help.  
- **Step4 : Obtain *Deepo***  
  A series of Docker images (and their generator) that allows you to quickly set up your deep learning research environment. See [*Deepo*](https://github.com/ufoym/deepo) to get more details. Run this and turn on the **port 6006** for the tensorboard:
  
  ```bash
  $ nvidia-docker pull ufoym/deepo  
  $ nvidia-docker run 
  ```
  enter the bash of *Deepo*, run pip to install the rest： 
  ```bash
  $ pip install gensim rouge tflearn tqdm
  ```
* **Step5: Train the model and predict**  

  ```bash
  $ python train_model.py
  ```

## Model details

* **Structure NN-SE**  

<img src="https://drive.google.com/uc?export=download&id=1JrGt0VcqR_QmTzPG3OAEgOeoQYXUeMXp" width = "450" height = "450" alt="sentence_model" align=center />  

* **Sentence extractor**  
  Here is the single step of the customzied LSTM with a score layer.
  ```python
  def lstm_single_step(self, St, At, h_t_minus_1, c_t_minus_1, p_t_minus_1):
      p_t_minus_1 = tf.reshape(p_t_minus_1, [-1, 1])
      # Xt = p_t_minus_1 * St
      Xt = tf.multiply(p_t_minus_1, St)
      # dropout
      Xt = tf.nn.dropout(Xt, keep_prob=self.dropout_keep_prob)
      # compute the gate of input, forget, output 
      i_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_i) + tf.matmul(h_t_minus_1, self.U_i) + self.b_i)
      f_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_f) + tf.matmul(h_t_minus_1, self.U_f) + self.b_f)
      c_t_candidate = tf.nn.tanh(tf.matmul(Xt, self.W_c) + tf.matmul(h_t_minus_1, self.U_c) + self.b_c)
      c_t = f_t * c_t_minus_1 + i_t * c_t_candidate
      o_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_o) + tf.matmul(h_t_minus_1, self.U_o) + self.b_o)
      h_t = o_t * tf.nn.tanh(c_t)
      # compute prob
      with tf.name_scope("Score_Layer"):
          concat_h = tf.concat([At, h_t], axis=1)
          concat_h_dropout = tf.nn.dropout(concat_h, keep_prob=self.dropout_keep_prob)
          score = tf.layers.dense(concat_h_dropout, 1, activation=tf.nn.tanh, name="score", reuse=tf.AUTO_REUSE)
      # activation and normalization
      p_t = self.sigmoid_norm(score)
      return h_t, c_t, p_t
  ```
* **Curriculum learning**  
  Actually new to curriculum learning， just simply connect the weight of the true labels and those predicted with the rate of steps.
  ```python
  def weight_control(self, time_step, p_t):
      # curriculum learning control the weight between true labels and those predicted
      labels = tf.cast(self.input_y1[:,time_step:time_step+1], dtype=tf.float32)
      total_step = tf.cast(self.cur_step, dtype=tf.float32)
      global_step = tf.cast(self.global_step, dtype=tf.float32)
      weight = tf.divide(global_step, total_step)
      p_t_curr = (1 - weight) * labels + weight * p_t
      return p_t_curr
  ```  
  
* **Loss function**  
  Coding the loss function manually instead of using the function *tf.losses.sigmoid_cross_entropy* cause the logits is between 0 and 1  with sigmoid activation and normalization already.  
  ```python
  # loss:z*-log(x)+(1-z)*-log(1-x)
  # z=0 --> loss:-log(1-x)
  # z=1 --> loss:-log(x)
  with tf.name_scope("loss_sentence"):
      logits = tf.convert_to_tensor(self.logits)
      labels = tf.cast(self.input_y1, logits.dtype)
      zeros = tf.zeros_like(labels, dtype=labels.dtype)
      ones = tf.ones_like(logits, dtype=logits.dtype)
      cond  = ( labels > zeros )
      logits_ = tf.where(cond, logits, ones-logits)
      logits_log = tf.log(logits_)
      losses = -logits_log
      loss = tf.reduce_sum(losses, axis=1)
      loss = tf.reduce_mean(loss)     
  ```


## Performance

* **probability for the sentences**

* **loss**

* **evaluation**

## Discuss

## TODO list

## Credits
