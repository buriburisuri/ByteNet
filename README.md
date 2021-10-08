# ByteNet - Fast Neural Machine Translation
A tensorflow implementation of French-to-English machine translation using DeepMind's ByteNet 
from the paper [Nal et al's Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099).
This paper proposed the fancy method which replaced the traditional RNNs with conv1d dilated and causal conv1d, 
and they achieved fast training and state-of-the-art performance on character-level translation. 

The architecture ( from the paper )
<p align="center">
  <img src="https://raw.githubusercontent.com/buriburisuri/ByteNet/master/png/architecture.png" width="1024"/>
</p>

## Version

Current Version : __***0.0.0.2***__

## Dependencies ( VERSION MUST BE MATCHED EXACTLY! )

1. [tensorflow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#pip-installation) == 1.0.0
1. [sugartensor](https://github.com/buriburisuri/sugartensor) == 1.0.0.2
1. [nltk](http://www.nltk.org/install.html) == 3.2.2

## Datasets

I've used NLTK's comtrans English-French parallel corpus for convenience.  You can easily download it as follows:
<pre><code>
python
>>>> import nltk
>>>> nltk.download_shell()
NLTK Downloader
---------------------------------------------------------------------------
    d) Download   l) List    u) Update   c) Config   h) Help   q) Quit
---------------------------------------------------------------------------
Downloader> d

Download which package (l=list; x=cancel)?
  Identifier> comtrans
  
</code></pre>

## Implementation differences from the paper.

1. I've replaced the Sub Batch Normal with [Layer Normalization](https://arxiv.org/abs/1607.06450) for convenience.
1. No bags of characters applied for simplicity.
1. Latent dimension is 400 because Comtrans corpus in NLTK is small. ( 892 in the paper )
1. Generation code not optimized.

## Training the network

Execute
<pre><code>
python train.py
</code></pre>
to train the network. You can see the result ckpt files and log files in the 'asset/train' directory.
Launch tensorboard --logdir asset/train/log to monitor training process.

I've trained this model on a single Titan X GPU during 10 hours until 50 epochs. 
If you don't have a Titan X GPU, reduce batch_size in the train.py file from 16 to 8.  

## Translate sample French sentences
 
Execute
<pre><code>
python translate.py
</code></pre>
to translate sample French sentences to English. The result will be printed on the console. 

## Sample translations

The result looks messy but promising. 
Though Comtrans corpus in NLTK is very small(in my experiment only 17,163 pairs used), 
the model have learned English words structures and syntax by character level.  
I think that the translation accuracy will be better if we use big corpus. 
  

| French (sources) | English (translated by ByteNet) | English (translated by Google translator) |
| :------------- | :------------- | :------------- |
| Et pareil phénomène ne devrait pas occuper nos débats ? | And they applied commitments have been satisfied ? | And such a phenomenon should not occupy our debates? |
| Mais nous devons les aider sur la question de la formation . | However , we must addruss that climate condition . | But we need help on the issue of training. |
| Les videurs de sociétés sont punis . | The existing considerations in the coming years ago . | Corporate bouncers are punished. |
| Après cette période , ces échantillons ont été analysés et les résultats illustrent bien la quantité de dioxine émise au cours des mois écoulés . | According to the relevant continent with the intentions and for all , the points of building situation by the directive butchers . | After this period, the samples were analyzed and the results illustrate the amount of dioxins emitted during the past months. |
| Merci beaucoup , Madame la Commissaire . | Thank you very much for the Commissioner against this perfect . | Thank you very much, Commissioner. |
| Le Zimbabwe a beaucoup à gagner de l ' accord de partenariat et a un urgent besoin d ' aide et d ' allégement de la dette . | The AIDR problem is carried out corperation in the waken home after a peaceful future and not have their different parts . | Zimbabwe has much to gain from the Partnership Agreement and urgently needs aid and debt relief. |
| Le gouvernement travailliste de Grande-Bretagne a également des raisons d ' être fier de ses performances . | The Larning wants to have a former colleague with the United States is indeed all of the population . | The Labour government in Britain also has reason to be proud of its performance. |
| La plupart d' entre nous n' a pas l' intention de se vanter des 3 millions d' euros . | Most of us here would not wish to boast about EUR 3 million . | Most of us do not have the intention to boast of 3 million euros. |
| Si le Conseil avait travaillé aussi vite que ne l' a fait M. Brok , nous serions effectivement bien plus avancés . | If the Council had worked as quickly as Mr Brok then have been done and general support . | If the Council had worked as quickly as did the did Mr Brok, we would indeed well advanced. |
| Le deuxième thème important concerne la question de la gestion des contingents tarifaires . | The second important area is the issue of managing tariff quotas .| The second important issue concerns the question of the management of tariff quotas. |


## pre-trained models

You can translate French sentences to English sentences with the pre-trained model on the Comtrans corpus in NLTK. 
Extract [the following zip file](https://drive.google.com/file/d/0B3ILZKxzcrUyeXBVeVZoWlN5XzA/view?usp=sharing&resourcekey=0-XtdX-roA29Z5KtpVPrKChA) in 'asset/train'.
And try another sample French sentences in the 'translate.py' file.  
 
## Other resources

1. [ByteNet language model tensorflow implementation](https://github.com/paarthneekhara/byteNet-tensorflow)

## My other repositories

1. [SugarTensor](https://github.com/buriburisuri/sugartensor)
1. [EBGAN tensorflow implementation](https://github.com/buriburisuri/ebgan)
1. [Timeseries gan tensorflow implementation](https://github.com/buriburisuri/timeseries_gan)
1. [Supervised InfoGAN tensorflow implementation](https://github.com/buriburisuri/supervised_infogan)
1. [AC-GAN tensorflow implementation](https://github.com/buriburisuri/ac-gan)
1. [SRGAN tensorflow implementation](https://github.com/buriburisuri/SRGAN)

# Authors
Namju Kim (buriburisuri@gmail.com) at Jamonglabs Co., Ltd.
