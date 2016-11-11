# ByteNet - Machine Translation
A tensorflow implementation of French-to-English machine translation using DeepMind's ByteNet 
from the paper [Nal et al's Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099).
This paper proposed the fancy method which replaced the traditional RNNs with conv1d dilated and causal conv1d, 
and they attained state-of-the-art performance on character-level language modeling. 

The architecture ( from the paper )
<p align="center">
  <img src="https://raw.githubusercontent.com/buriburisuri/ByteNet/master/png/architecture.png" width="800"/>
</p>

## Dependencies

1. tensorflow >= rc0.11
1. [sugartensor](https://github.com/buriburisuri/sugartensor) >= 0.0.1.7
1. [nltk](http://www.nltk.org/install.html) >= 3.0

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
2. No bags of characters applied for simplicity.
3. Latent dimension is 500 because comtrans corpus in NLTK is small. ( 892 in the paper )

## Training the network

Execute
<pre><code>
python train.py
</code></pre>
to train the network. You can see the result ckpt files and log files in the 'asset/train' directory.
Launch tensorboard --logdir asset/train/log to monitor training process.

## Translate sample French sentences
 
Execute
<pre><code>
python translate.py
</code></pre>
to translate sample French sentences to English. The result will be printed on the console. 

## Sample translations

| French (sources) | English translated by ByteNet | English translated by Google translator |
| :------------- | :------------- | :------------- |
| Et pareil phénomène ne devrait pas occuper nos débats ? | to-do | And such a phenomenon should not occupy our debates? |
| Mais nous devons les aider sur la question de la formation . | to-do | But we need help on the issue of training. |
| Les videurs de sociétés sont punis . | to-do | Corporate bouncers are punished. |
| Après cette période , ces échantillons ont été analysés et les résultats illustrent bien la quantité de dioxine émise au cours des mois écoulés . | to-do | After this period, the samples were analyzed and the results illustrate the amount of dioxins emitted during the past months. |
| Merci beaucoup , Madame la Commissaire . | to-do | Thank you very much, Commissioner. |
| Le Zimbabwe a beaucoup à gagner de l ' accord de partenariat et a un urgent besoin d ' aide et d ' allégement de la dette . | to-do | Zimbabwe has much to gain from the Partnership Agreement and urgently needs aid and debt relief. |
| Le gouvernement travailliste de Grande-Bretagne a également des raisons d ' être fier de ses performances . | to-do | The Labour government in Britain also has reason to be proud of its performance. |
| La plupart d' entre nous n' a pas l' intention de se vanter des 3 millions d' euros . | to-do | Most of us do not have the intention to boast of 3 million euros. |
| Si le Conseil avait travaillé aussi vite que ne l' a fait M. Brok , nous serions effectivement bien plus avancés . | to-do | If the Council had worked as quickly as did the did Mr Brok, we would indeed well advanced. |
| Le deuxième thème important concerne la question de la gestion des contingents tarifaires . | to-do | The second important issue concerns the question of the management of tariff quotas. |

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