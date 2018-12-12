
TextSumma
=
Just give it a shot for reproducing the ACL 2016 paper [*Neural Summarization by Extracting Sentences and Words*](https://arxiv.org/abs/1603.07252). Find original code [*here*](https://github.com/cheng6076/NeuralSum).

## Quick Start
- **Step1 : Obtain datasets**  
   Go [*here*](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) to download the corpus and get the scripts of ***one-billion-word-language-modeling-benchmark*** for training the word vectors. Run this and see more [*details*](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark/blob/master/README.corpus_generation):
   ```bash
   $ tar --extract -v --file ../statmt.org/tar_archives/training-monolingual.tgz --wildcards training-monolingual/news.20??.en.shuffled
   $ ./scripts/get-data.sh
   ```
   The dataset ***cnn-dailymail*** with highlights in this paper offered by the authors is in [*here*](https://docs.google.com/uc?id=0B0Obe9L1qtsnSXZEd0JCenIyejg&export=download) and vocab in this repository.  
- **Step2 : Preprocess**  
  Run this script to training the word vectors in the dataset ***one-billion-word-language-modeling-benchmark***:  
  ```bash
  $ python train_w2v.py './one-billion-word-benchmark' 'output_gensim_model' 'output_word_vector'
  ```
  Run this script to extract the sentences, labels and entitys in the dataset ***cnn-dailymail*** and get them pickled:
  ```bash
  $ python prepro.py './source_dir/' './target_dir/'
  ``` 
- **Step3 : Install nvidia-docker**  
  Go for GPUs acceleration. See [*installation*](https://github.com/NVIDIA/nvidia-docker) to get more information for help.  
- **Step4 : Obtain *Deepo***  
  A series of Docker images (and their generator) that allows you to quickly set up your deep learning research environment. See [*Deepo*](https://github.com/ufoym/deepo) to get more details. Run this and turn on the **port 6006** for the tensorboard:
  
  ```bash
  $ nvidia-docker pull ufoym/deepo  
  $ nvidia-docker run -p 0.0.0.0:6006:6006 -it -v /home/usrs/yourdir:/data ufoym/deepo env LANG=C.UTF-8 bash
  ```
  enter the bash of *Deepo*, run pip to install the rest: 
  ```bash
  $ pip install gensim rouge tflearn tqdm
  ```
* **Step5: Train the model and predict**

  Please add option **-h** to get more help in flag settings.  
  ```bash
  $ python train_model.py
  $ python predict_model.py
  ```
 * **Requirements**:
 Python3.6 Tensorflow 1.8.0

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
      start = tf.cast(self.cur_step_start, dtype=tf.float32)
      end = tf.cast(self.cur_step_end, dtype=tf.float32)
      global_step = tf.cast(self.global_step, dtype=tf.float32)
      weight = tf.divide(tf.subtract(global_step, start), tf.subtract(end, start))
      merge = (1. - weight) * labels + weight * p_t
      cond = tf.greater(start, global_step)
      p_t_curr = tf.cond(cond, lambda:labels, lambda:merge)
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
      losses *= self.mask
      loss = tf.reduce_sum(losses, axis=1)
      loss = tf.reduce_mean(loss)     
  ```


## Performance

* **Probability for the sentences in several timesteps**

<img src="https://drive.google.com/uc?export=download&id=18x5jRQ6VWke9PHSRLORxcIhSDjR3vvNx" width = "1000" height = "600" alt="sentence_model" align=center />

* **Training loss**

<img src="https://drive.google.com/uc?export=download&id=1yJ4NQZ1IsEYc-kPDJ3RN8E9tryPMlVQq" width = "450" height = "200" alt="sentence_loss" align=center />

* **Figure**  
Some results seems to be nice.
```json
{
  "entity": {
    "@entity31": "Jason Kernick", 
    "@entity1": "Manchester", 
    "@entity9": "Ashton Canal", 
    "@entity46": "Environment Agency", 
    "@entity44": "Etihad stadium", 
    "@entity45": "Manchester City", 
    "@entity115": "Easter Sunday", 
    "@entity85": "Clayton", 
    "@entity66": "Richard Kernick", 
    "@entity109": "Etihad", 
    "@entity137": "Greater Manchester Fire and Rescue Service", 
    "@entity136": "Salford"
  }, 
  "abstract": [
    "the @entity9 became filled with heavy suds due to a 6ft wall of foam created by fire crews tackling a blaze", 
    "the fire at a nearby chemical plant saw water from fire service mix with detergents that were being stored there", 
    "the foam covered a 30 metre stretch of the canal near @entity45 's @entity44 in @entity85"
  ], 
  "article": [
    "a @entity1 canal was turned into a giant bubble bath after fire crews tackling a nearby chemical plant blaze saw their water mix with a detergent creating a six foot wall of foam", 
    "the @entity9 was filled with heavy suds which appeared after a fire at an industrial unit occupied by a drug development company", 
    "it is believed that the water used by firefighters to dampen down the flames mixed with the detergent being stored in the burning buildings", 
    "now the @entity46 have launched an investigation to assess if the foam has impacted on wildlife after concerns were raised for the safety of fish in the affected waters", 
    "a spokesman for the agency said : ' @entity46 is investigating after receiving reports of foam on a 30 metre stretch of the @entity9 , @entity1", 
    "' initial investigations by @entity46 officers show that there appears to have been minimal impact on water quality , but our officers will continue to monitor and respond as necessary", 
    "@entity66 takes a picture on his mobile phone of his boat trying to negotiate a lock and the foam , which ran into the @entity9 a cyclist takes a picture on his mobile phone as the foam comes up on to the cycle path", 
    "the @entity46 are investigating to assess of the foam has harmed any wildlife the foam reached as high as six foot in some places and covered a 30 metre stretch along the water in the @entity85 area of @entity1 ' we are working with the fire service and taking samples of the foam to understand what it is made of , and what impact it may have on local wildlife in and around the canal", 
    "' at the height of the blaze on sunday afternoon , which caused the foam , up to 50 firefighters were tackling the fire and police were also forced to wear face masks", 
    "families in east @entity1 were urged to say indoors after a blast was reported at the industrial unit , which is just a few hundred yards from the @entity45 training ground on the @entity109 campus", 
    "the fire at the chemical factory next to @entity45 's @entity44 send a huge plume of smoke across the city on @entity115 police wearing face masks went around neighbouring streets with loudspeakers urging people to stay inside while the fire raged police officers also told children on bikes and mothers pushing prams near the scene to go home and went around neighbouring streets with loudspeakers urging people to stay inside", 
    "a huge plume of smoke also turned the sky black and could be seen right across the city and even into @entity136", 
    "according to @entity137 , the fire was fueled by wooden pallets and unidentified chemicals but an investigation into the cause of the fire is still ongoing ."
  ], 
  "label": [0, 1, 4, 7, 10], 
  "score": [
    [10, 0.6629698276519775, "the fire at the chemical factory next to @entity45 's @entity44 send a huge plume of smoke across the city on @entity115 police wearing face masks went around neighbouring streets with loudspeakers urging people to stay inside while the fire raged police officers also told children on bikes and mothers pushing prams near the scene to go home and went around neighbouring streets with loudspeakers urging people to stay inside"], 
    [0, 0.6484572291374207, "a @entity1 canal was turned into a giant bubble bath after fire crews tackling a nearby chemical plant blaze saw their water mix with a detergent creating a six foot wall of foam"], 
    [7, 0.5045493841171265, "the @entity46 are investigating to assess of the foam has harmed any wildlife the foam reached as high as six foot in some places and covered a 30 metre stretch along the water in the @entity85 area of @entity1 ' we are working with the fire service and taking samples of the foam to understand what it is made of , and what impact it may have on local wildlife in and around the canal"], 
    [1, 0.45766133069992065, "the @entity9 was filled with heavy suds which appeared after a fire at an industrial unit occupied by a drug development company"], 
    [4, 0.3478981852531433, "a spokesman for the agency said : ' @entity46 is investigating after receiving reports of foam on a 30 metre stretch of the @entity9 , @entity1"], 
    [3, 0.3398599326610565, "now the @entity46 have launched an investigation to assess if the foam has impacted on wildlife after concerns were raised for the safety of fish in the affected waters"], 
    [8, 0.3396754860877991, "' at the height of the blaze on sunday afternoon , which caused the foam , up to 50 firefighters were tackling the fire and police were also forced to wear face masks"], 
    [6, 0.32800495624542236, "@entity66 takes a picture on his mobile phone of his boat trying to negotiate a lock and the foam , which ran into the @entity9 a cyclist takes a picture on his mobile phone as the foam comes up on to the cycle path"], 
    [9, 0.29064181447029114, "families in east @entity1 were urged to say indoors after a blast was reported at the industrial unit , which is just a few hundred yards from the @entity45 training ground on the @entity109 campus"], 
    [2, 0.25459226965904236, "it is believed that the water used by firefighters to dampen down the flames mixed with the detergent being stored in the burning buildings"], 
    [5, 0.2020452618598938, "' initial investigations by @entity46 officers show that there appears to have been minimal impact on water quality , but our officers will continue to monitor and respond as necessary"], 
    [12, 0.05926991254091263, "according to @entity137 , the fire was fueled by wooden pallets and unidentified chemicals but an investigation into the cause of the fire is still ongoing ."], 
    [11, 0.05400165915489197, "a huge plume of smoke also turned the sky black and could be seen right across the city and even into @entity136"]
  ]
}

```
This one remains a little complicated.
```json
{
  "entity": {
    "@entity27": "Belichick", 
    "@entity24": "Hiss", 
    "@entity80": "Gisele Bündchen", 
    "@entity97": "Sport Illustrated", 
    "@entity115": "Julian Edelman", 
    "@entity84": "Washington", 
    "@entity86": "Seattle Seahawks", 
    "@entity110": "Massachusetts US Senator", 
    "@entity3": "Patriots", 
    "@entity2": "Super Bowl", 
    "@entity0": "Obama", 
    "@entity4": "White House", 
    "@entity8": "South Lawn", 
    "@entity56": "Boomer Esiason", 
    "@entity111": "Linda Holliday", 
    "@entity75": "Donovan McNabb", 
    "@entity96": "Las Vegas", 
    "@entity30": "Chicago", 
    "@entity33": "Boston", 
    "@entity102": "Rob Gronkowski", 
    "@entity99": "CBS", 
    "@entity98": "Les Moonves", 
    "@entity108": "Bellichick", 
    "@entity109": "John Kerry", 
    "@entity95": "Floyd Mayweather Jr.", 
    "@entity94": "Manny Pacquiao", 
    "@entity117": "Danny Amendola", 
    "@entity62": "Bush Administration", 
    "@entity44": "Bob Kraft", 
    "@entity47": "Super Bowl MVP", 
    "@entity68": "Showoffs", 
    "@entity66": "US Senator", 
    "@entity67": "White House Correspondents dinner", 
    "@entity113": "George W. Bush", 
    "@entity48": "Brady"
  }, 
  "abstract": [
    "@entity48 cited ' prior family commitments ' in bowing out of meeting with @entity0", 
    "has been to the @entity4 to meet president @entity113 for previous @entity2 wins"
  ], 
  "article": [
    "president @entity0 invited the @entity2 champion @entity3 to the @entity4 on thursday - but could n't help but get one last deflategate joke in", 
    "the president opened his speech on the @entity8 by remarking ' that whole ( deflategate ) story got blown out of proportion , ' referring to an investigation that 11 out of 12 footballs used in the afc championship game were under - inflated", 
    "but then came the zinger : ' i usually tell a bunch of jokes at these events , but with the @entity3 in town i was worried that 11 out of 12 of them would fall flat", 
    "coach @entity27 , who is notoriously humorless , responded by giving the president a thumbs down", 
    "@entity0 was flanked by @entity27 and billionaire @entity3 owner @entity44", 
    "missing from the occasion , though was the @entity47 and the team 's biggest star - @entity48", 
    "a spokesman for the team cited ' prior family commitments ' as the reason @entity48 , 37 , did n't attend the ceremony", 
    "sports commentators , including retired football great @entity56 , speculated that @entity48 snubbed @entity0 because he 's from the ' wrong political party", 
    "' the superstar athlete has been to the @entity4 before", 
    "he does have three other @entity2 rings , afterall", 
    "but all the prior championships were under the @entity62", 
    "february 's win was the first for the @entity3 since @entity0 took office", 
    "@entity48 has also met @entity0 at least once before , as well", 
    "he was pictured with the then - @entity66 at the 2005 @entity67", 
    "@entity68 : the @entity3 gathered the team 's four @entity2 trophies won under coach @entity27 ( right , next to president @entity0 )", 
    "@entity48 won his fourth @entity2 ring in february - and his first since president @entity0 took office @entity48 met president @entity0 at least once", 
    "he is pictured here with the then - @entity66 and rival quarterback @entity75 in 2005 it 's not clear what @entity48 's prior commitment was", 
    "his supermodel wife @entity80 , usually active on social media , gives no hint where the family is today if not in @entity84", 
    "@entity48 led the @entity3 to his fourth @entity2 victory in february after defeating the @entity86 28 - 24", 
    "despite his arm and movement being somewhat diminished by age , @entity48 's leadership and calm under pressure also won him @entity47 - his third", 
    "whatever is taking up @entity48 's time this week , he made time next week to be ringside at the @entity95 - @entity94 fight in @entity96 next weekend", 
    "according to @entity97 , @entity48 appealed directly to @entity99 president @entity98 for tickets to the much - touted matchup", 
    "@entity3 tight end @entity102 could n't help but mug for the camera as the commander in chief gave a speech @entity0 walks with billionaire @entity3 owner @entity44 and coach @entity108 to the speech secretary of state @entity109 , a former @entity110 , greets @entity27 's girlfriend @entity111 at the ceremony @entity48 went to the @entity4 to meet president @entity113 after winning the @entity2 in 2005 and in 2004", 
    "he 's not going to be there this year @entity3 players @entity115 and @entity117 snap pics in the @entity4 before meeting president @entity0 on thursday"
  ], 
  "label": [0, 6, 7, 12, 14, 15, 18, 22, 23], 
  "score": [
    [1, 0.8683828115463257, "the president opened his speech on the @entity8 by remarking ' that whole ( deflategate ) story got blown out of proportion , ' referring to an investigation that 11 out of 12 footballs used in the afc championship game were under - inflated"], 
    [0, 0.8339700102806091, "president @entity0 invited the @entity2 champion @entity3 to the @entity4 on thursday - but could n't help but get one last deflategate joke in"], 
    [22, 0.7730730772018433, "@entity3 tight end @entity102 could n't help but mug for the camera as the commander in chief gave a speech @entity0 walks with billionaire @entity3 owner @entity44 and coach @entity108 to the speech secretary of state @entity109 , a former @entity110 , greets @entity27 's girlfriend @entity111 at the ceremony @entity48 went to the @entity4 to meet president @entity113 after winning the @entity2 in 2005 and in 2004"], 
    [14, 0.7569227814674377, "@entity68 : the @entity3 gathered the team 's four @entity2 trophies won under coach @entity27 ( right , next to president @entity0 )"], 
    [15, 0.6214166879653931, "@entity48 won his fourth @entity2 ring in february - and his first since president @entity0 took office @entity48 met president @entity0 at least once"], 
    [18, 0.4963235855102539, "@entity48 led the @entity3 to his fourth @entity2 victory in february after defeating the @entity86 28 - 24"], 
    [16, 0.45303720235824585, "he is pictured here with the then - @entity66 and rival quarterback @entity75 in 2005 it 's not clear what @entity48 's prior commitment was"], 
    [5, 0.4204302430152893, "missing from the occasion , though was the @entity47 and the team 's biggest star - @entity48"], 
    [7, 0.41678884625434875, "sports commentators , including retired football great @entity56 , speculated that @entity48 snubbed @entity0 because he 's from the ' wrong political party"], 
    [20, 0.4135805070400238, "whatever is taking up @entity48 's time this week , he made time next week to be ringside at the @entity95 - @entity94 fight in @entity96 next weekend"], 
    [6, 0.3958345353603363, "a spokesman for the team cited ' prior family commitments ' as the reason @entity48 , 37 , did n't attend the ceremony"], 
    [4, 0.37495893239974976, "@entity0 was flanked by @entity27 and billionaire @entity3 owner @entity44"], 
    [21, 0.3466879427433014, "according to @entity97 , @entity48 appealed directly to @entity99 president @entity98 for tickets to the much - touted matchup"], 
    [19, 0.3316606283187866, "despite his arm and movement being somewhat diminished by age , @entity48 's leadership and calm under pressure also won him @entity47 - his third"], 
    [2, 0.29267093539237976, "but then came the zinger : ' i usually tell a bunch of jokes at these events , but with the @entity3 in town i was worried that 11 out of 12 of them would fall flat"], 
    [23, 0.27186375856399536, "he 's not going to be there this year @entity3 players @entity115 and @entity117 snap pics in the @entity4 before meeting president @entity0 on thursday"], 
    [11, 0.26710671186447144, "february 's win was the first for the @entity3 since @entity0 took office"], 
    [17, 0.17511016130447388, "his supermodel wife @entity80 , usually active on social media , gives no hint where the family is today if not in @entity84"], 
    [3, 0.16352418065071106, "coach @entity27 , who is notoriously humorless , responded by giving the president a thumbs down"], 
    [13, 0.14906153082847595, "he was pictured with the then - @entity66 at the 2005 @entity67"], 
    [12, 0.1384015828371048, "@entity48 has also met @entity0 at least once before , as well"], 
    [10, 0.07186555117368698, "but all the prior championships were under the @entity62"], 
    [8, 0.07148505747318268, "' the superstar athlete has been to the @entity4 before"], 
    [9, 0.035264041274785995, "he does have three other @entity2 rings , afterall"]
  ]
}
```
Pls get more predicted results from [*here*](https://drive.google.com/open?id=1cXrR1kY-tlxArB-F9FSZba2T2RscAYVS)

## Discuss

- Tuning the learning rate.
- Freeze the weights of embedding for several steps or not.
- Choose a proper step range for shift the value gradually to the probability predicted by the model.
- The initialization of the weights and bias.
- Find a proper way to evaluate while training.(just observe the loss in validation with early stop by my way)

## TODO list
- NN-WE word extractor remain to be done.  
- Remain oov and ner problems while using raw data. 
- Using Threading and Queues in tensorflow to load the batch.  

## Credits
- Thanks for the authors of the paper.
- Borrow some code from [*text_classification*](https://github.com/brightmart/text_classification) and learn a lot.
- A great job [*pointer-generator*](https://github.com/abisee/pointer-generator) in text summarization that should be appreciated.
