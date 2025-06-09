---
layout: post
title:  "Имплементациjа На Трансформер Архитектурата За Македонско-Англиски Превод На Реченици"
date:   2025-06-06 09:32:32 +0200
categories: basics
author: Стефан Најдовски, Христијан Горков
---

<br>

---

<br>

[![Run on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/najdovski-stefan/Donka-v1/blob/main/Donka_v1_Inference_seq2seq_mk_en-GOOGLE-COLAB.ipynb)

<br>

---

<br>


<div style="background-color: #ff0000; color: #ffffff; border: 1px solid #ffeeba; padding: 1rem; border-radius: 6px; margin: 1rem 0;">
    <strong>⚠️  Блогот е во процес на пишување и валидација:</strong>
    <br>
    <br>
        Хаотична Драфт верзија
        Ве молиме вратете се подоцна
    <br>
    <br>

    Ви благодариме

</div>

---

Блогов што го читате е резултат на <b>работните групи по вештачка интелигенција</b>, каде што ние студентите <b>Стефан Најдовски</b> и <b>Христијан Горков</b>, запишани на прв циклус студии на [Факултетот за Информатички и Комункациски Технологии - Битола](https://fikt.uklo.edu.mk/) под менторство на [Проф д-р Костандина Вељановска](https://fikt.uklo.edu.mk/prof-d-r-kostandina-veljanovski/).
и асистентот [М-р Дарко Пајковски]((https://fikt.uklo.edu.mk/darko-pajkovski/)).

Нашата тема на обработка ќе ви ја претставиме детално, низ илустрации а резулатот од нашето мини истражување е <b>мал модел кој може да преведува македонски текст</b>.

<br>

---

<br>

# Содржина

1. [Вовед](#0-вовед)
2. [Што е токен?](#1-токен) (Token)
3. Податоци за тренирање (Data Set)
4. [Токенизирање]() (Tokenization)
5. [SentencePiece библиотека]()
6. Token Eembeddings
7. Секвенца во Секвенца
8. Лимитации

---

<br>



## 1. Вовед



Целта на овој блог е со <b>двојна природа</b>, <b>првенствено е наменета да научиме како функционира Transformer архитектурата</b>, делот кој ја направи науката за вештачки неуронски мрежи <b>посериозна, практична и достапна за секого</b>, најголема примена има во [Large Language Models](https://www.ibm.com/think/topics/large-language-models) како [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), Claude, Mistral, LLAMA...

<b>Втората цел ни е да ви претставиме мал модел</b> кој знае да преведува кратки реченици од македонски јазик на англиски јазик.

На крајот има и <b>demo</b> од проектот, што може да се користи практично, со одредени лимитации.

Секако нашата имплементација тука ќе биде <b>многу пати поедноставна </b>, но есенцијално идејата е иста, скоро целосно базирана врз оригиналниот труд [Attention is All You Need](https://arxiv.org/pdf/1706.03762).

Целта <b>нема</b> да биди:  Generative Pretrained Transformer (GPT), имплементација на целосен Large Language Model (LLM), е далеку покомплицирано со теорија и уште потешко за имплеметнација, сакавме да бидиме јасни за која е целта.

Она што ќе го имплементираме и објасниме, ќе биди:

<b>Sequence to Sequence Vanilla Transformer</b> или на кратко <b>Seq2Seq Transformer</b>.

Ќе гледаме да балансираме со технички жаргони и да објасниме интуитивно и со примери за секој да може да не следи.

Ви препорачуваме да имате некоја блага основа за полесно следење, за тоа што е <b>Neural Network</b> и што е тоа <b>Natural Language Processing</b>, исто препорачливо е познавање на <b>основи на веројатност и статистика</b>.

Доколку сакате да научите или сте љубопитни и жедни за знаење слободно погледнете:
[Корисни ресурси за почетници](#корисна-литература-за-почетници-и-за-љубопитните)

<br>


<div style="background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; padding: 1rem; border-radius: 6px; margin: 1rem 0;">
    <strong>⚠️ Мало Предупредување:</strong>

    <br>
    <br>

        Овој блог е пишуван од студенти, сè уште ги проучуваме детално сите идеи презентирани на блогот

  <br>
  <br>

  Сигурни сме дека имаме некаде грешка. Очекувајте грешки како граматички и нестандарден јазик.

  <br>
  <br>

  <b>Отворени сме на конструктивни критики</b>

  <br>
  <br>
  Доколку пронајдите какви било грешки слободно контактирајте не.

  <br>
  <br>
  Благодариме :)
</div>

<br>

---

<br>

## 1. Токен

Овој збор кај нас би бил преведен како <b>лексема</b> или <b>жетон</b>, во англиската (програмерската) литература e дефиниран како <b>атомична (неделива) единица за репрезентација нa текст.</b> (искрено не сме сигурни дали го имаат истото значење на македонски или општо во лингвистиката).

Должината на оваа единица е <b>произволна</b> и зависи од проблемот што сакаме да го решиме.

Се сретнува во повеќе должини:
  - <b>Реченица</b> (пример: Здраво Македонијо!)
  - <b>Дел од збор</b> (пример во вистински јазик би биле слоговите).
  - <b>Збор</b>(пример: <b>здраво</b>).
  - <b>Карактер</b> (пример: <b>а</b>).
  - <b>Бајт</b> (пример: <b>ASCII</b> или <b>UTF</b> енкодиран карактер).

![granularnost](/assets/images/granularnost.png)


<br>

---

<br>

## 3.Речник

## 4. Токенизирање

Е процесот на претворање на текст (во нашиот случај македонска кирилица) во токени, со тоа што подоцна истите тие ќе бидат претставени како <b>вектори</b> за моделот да може да ги процесира.

Типови на токенајзери:

- со правила (Rule-based) (токен-збор,токен-буква), најнеоптимален.

- Научен (Learned) тип:

- Во Научените токенизатори спаѓаат: Byte-Pair Encoding (BPE), WordPiece и Униграм.

![tokenizator](/assets/images/tokenizator.png)

 - ние го користиме <b>Unigram</b>, со помош на <b>sentencepiece</b> библиотеката.



## 6. Dataset

За оние кои не се запознаени Data set <b>претставува колекција од податоци</b>, најчесто организирани во табела.

Изглед на нашата "табела":

```tsv
здраво  hello
ние сме студенти.  we are students.
јас сакам да учам.  I want to learn.
...
```

За тренирање на нашиот модел ние искористивме корпус кој е достапен на интернет, секако со дозвола на авторите кои можите да ги најдите [тука](#благодарност-до).

Податоците за тренирање ги зачувавме во формат наречен [Tab-Separated Value](https://www.loc.gov/preservation/digital/formats/fdd/fdd000533.shtml) или <b>TSV</b> на кратко, со помош на библиотеката pandas во Python.

Во <b>првата колона ги ставивме речениците на македонски јазик</b>, во <b>втората колона ги ставивме преведените реченици на англиски јазик</b>, дел од речениците беа преведени од почеток, остатокот од другите користевме Google Translate и локални LLM модели со техника на дестилација да враќа формат кој е прифатлив.

Валидација правевме со неколку примероци за квалитетот. Но дефинитивно сметаме дека не е најдобар начин за превод.

Како резултат добивме релативно мал data set од <b>480 илјади преведени реченици</b>.

Дистрибуција според должина на реченици:


Со помош на овие македонско-англиски парови го [трениравме моделот](2025-06-06-transformer-mk-en#Тренирање:).

## 7. Севенца во Секвенца (Sequence to Sequence).


[Seq2Seq]() се користи за обработка на природни јазици [NLP](https://mk.wikipedia.org/wiki/%D0%9E%D0%B1%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%BA%D0%B0_%D0%BD%D0%B0_%D0%BF%D1%80%D0%B8%D1%80%D0%BE%D0%B4%D0%BD%D0%B8_%D1%98%D0%B0%D0%B7%D0%B8%D1%86%D0%B8).

Во нашиот случај ќе го користиме за <b>превод од македонски на англиски</b>, мислиме дека е добар баланс помеѓу нешто што е <b> корисно да направиме за нашиот мајчин јазик</b> ,нешто што <b> не е само теорија</b> и нешто што <b>може да се научи</b>, три во едно :)

Пред да се појави трансформер архитектурата, <b>механизмите за "внимание"</b> биле ограничени со [GRU]() или [LSTM]() и користењето на [RNN-Recurrent Neural Networks]().


## 8. Positional embeddings

Оргиналната имплементација користи статични (фиксни) позициони вградувања.

За да се пресмета вредноста на едео позиционо вградување (3.5 во оригиналното истражувње).

Авторите ги користат функциите <b>синус и косинус</b> (наставниците и професорите беа во право, корисни се) :


{% raw %}
$$
PE(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
$$

$$
PE(\text{pos}, 2i+1) = \cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
$$
{% endraw %}

Енкодирањето зависи од 3 вредности:

- pos - позицијата на векторот
- i - индексот внатре во векторот
- d_model - димензијата на внесот


Позиционалните вградувања се користат за информирање на трансфомерот на која позиција се наоѓаат векторите за внес. Тие се додаваат на секоја вредност во векторот посебно,

![granularnost](/assets/images/visualize.png)

{% highlight python %}
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 10000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:])
{% endhighlight %}




## 9. Внимание
{% highlight python %}
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward:int = 512, dropout:float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
{% endhighlight %}

Концептот на <b>"Внимание"</b> е да го реши проблемот со преведување на текст, пред 20тина години овој проблем бил решаван со комплексни алгоритми кои имале бројни проблеми, наједноставниот проблем била самата <b>должина на речениците при превод</b>, тие се менуваат и стануваат уште по очигледни кога користиме јазици кои имаат различен начин на пишување, за повеќе околу проблемите од класичните начини на превод без корисење на неуронски мрежи можи да прочитате [тука](https://en.wikipedia.org/wiki/Statistical_machine_translation#Shortcomings).

### Клучеви, Вредности

{% raw %}
$$
Vnimanie(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$


{% endraw %}




## SentencePiece:

### Unigram:

е алгоритам за токенизација на под-зборови, каде што претпоставката е дека појавата на токен е <b>независна</b> од било кој од другите токени кои се појавиле претходно.


<br>

---

<br>

## Тренирање:

Под процесот тренирање се мисли учење на невронската мрежа (трансформерот) да преведува текст.

За овој чекор искористивме [Nvidia графичка RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) со 24 GB VRAM.

Моделот го трениравме 20 епохи, секоја епоха траеше околу еден час, после 20 епохи учење, моделот започна да покажува знаци на конвергенција (асимптотски паралелно со x оската).

![tokenizator](/assets/images/valtrainlossepoch.png)

За Validation Loss беше користено 10% од податоците, со фиксиран seed за репордукција (42). Причината за толку мал примерок е тоа што dataset-от е веќе мал, а 30% се премногу податоци да бидат надвор за валидација а не тренирање. Можеби и затоа се толку лоши резултатите.

Според табелата која е прикажана како <b>најдобар кандидат за инференца</b> се покажа епоха 18, со најниска вредност на валидација, тажно е што вредностите се над 3.

Претпоставуваме дека направивме грешка со learning-rate или грешка што искуството може само да ни ја открие, доколку некој поискусен знае слободно нека не корегира.

| Epoch | Train Loss | Validation Loss |
|-------|------------|-----------------|
| 1     | 4.593      | 4.0319          |
| 2     | 4.526      | 3.9444          |
| 3     | 4.056      | 3.6089          |
| 4     | 3.852      | 3.4798          |
| 5     | 3.792      | 3.4891          |
| 6     | 3.759      | 3.3927          |
| 7     | 3.729      | 3.3943          |
| 8     | 3.747      | 3.3443          |
| 9     | 3.715      | 3.3782          |
| 10    | 3.718      | 3.3259          |
| 11    | 3.678      | 3.2844          |
| 12    | 3.634      | 3.2948          |
| 13    | 3.603      | 3.2525          |
| 14    | 3.586      | 3.2174          |
| 15    | 3.559      | 3.1606          |
| 16    | 3.535      | 3.2004          |
| 17    | 3.527      | 3.1526          |
| 18    | 3.505      | 3.1024          |
| 19    | 3.529      | 3.1507          |
| 20    | 3.546      | 3.1696          |
| 21    | 3.536      | 3.1309          |
| 22    | 3.521      | 3.1388          |
| 23    | 3.538      | 3.1521          |


<br>

---

<br>


## Архитектура на моделот

| Параметри                        |             |
|----------------------------------|-------------|
| Македонски вокабулар             | 11,370      |
| Англиски вокабулар               | 8,257       |
| Големина на embedding            | 512         |
| Број на глави за внимание        | 8           |
| FFN скриен слој (FFN_HID_DIM)    | 512         |
| Големина на batch (BATCH_SIZE)   | 4           |
| Број на енкодер слоеви           | 3           |
| Број на декодер слоеви           | 3           |



<br>

---

<br>

## Резултати

Моделот е дефинитивно премал за реални апликации, со помали реченици добро се снаоѓа.



### Лимитации

1. <b>би се ставиле ние авторите како лимитација</b>, како почетници во оваа сфера, скоро 2 месеци лутавме по документации за да ги научиме основите, претпоставуваме дека имаме грешки при разбирање, имплементација, како и тренирање и валидација на моделот, но сметам дека <b>следната верзија на моделот ќе биде помоќна и корисна.</b>, дефинитивно има простор за подобрување.

2. Големината на Data-setот, 500 илјади пар реченици можеби звучат многу, но во пракса јазиците се покажуваат покомплексни од она што изгледаат на површина, дел од проблемот кои го воочивме е дека дури и моделите како GPT, Claude, LLAMA et al. ,<b> кои имаат десетици милијарди параметри</b>,кои се тренирани на <b>петабајти</b> податоци, имаат проблеми и потешкотии со македонскиот јазик, <b>не дека нашиот јазик не е богат, туку причината се недостаток на податоци во кванитет</b>, за вакви проекти се потребни квалитетни и квантитетни паралелни преводи.
Дефинитивно би направиле огромен отскок, доколку моделот беше трениран да речиме од 1 до 10 милиони квалитетни реченици.

3. Како трета лимитација е пристапот до тренирање на хардвер, за овој проект потрошивме околу 20 евра на тренирање на моделот (за пристап до RTX 4090).


### Простор за подобрување

Можеби е добро да се проба BERT архитектура или некоја сосема понова State Оf Тhe Аrt, која од една страна не бара голем број на податоци и не мора да се тренира од ништо, туку над неа со квалитетен Data-Set да се fine-тунира.


### Demo

[![Run on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/najdovski-stefan/Donka-v1/blob/main/Donka_v1_Inference_seq2seq_mk_en-GOOGLE-COLAB.ipynb)



<div style="background-color: #ff0000; color: #ffffff; border: 1px solid #ffeeba; padding: 1rem; border-radius: 6px; margin: 1rem 0;">
    <strong>⚠️  Предупредување:</strong>
    <br>
    <br>
    Моделот е лиценциран под
    <a href="https://www.creativecommons.org/licenses/by-nc/4.0/deed.en" target="_blank" style="color: #ffffff; text-decoration: underline;">
        Creative Commons Attribution Non Commercial 4.0
    </a>
    <br>
    <br>
    Моделот <b>МОЖЕ</b> да се користи за едукативни цели и истражувачки цели.
    <br>
    <br>
    Моделот <b>НЕ</b> може да биде користен во комерцијални продукти.
</div>

<br>

---

<br>


## Заклучок

Дефинитивно сметаме дека научивме и ја срушивме бариерата на илузија на моделите на Трансформери, бидејќи многу од нашите колеги и другари, имаат некој вид на страв и несигурност кон иднината, дека моделот наречен трансформер ќе започне да ги заменува со работни позиции, ние наспротив <b>оптимистички гледаме</b> дека потребата за луѓе и интелектуална работа ќе биде уште по интересна и достапна за секого. Оставаме на времето да ни покаже што понатака.



## Благодарност до:

Би сакале да изразиме посебна благодарност на следниве личности и институции, кои безусловно ни помогнаа и подржаја, без нив проектов не би постоел, од бесценети ресурси по македонски јазик, од песни, книги па се до вести, нашиот модел не би научил да преведи ниту една реченица, затоа би сакале да им се заблагодариме на следниве личности и институции:

- [Проф д-р Костандина Вељановска - ФИКТ - Битола](https://fikt.uklo.edu.mk/prof-d-r-kostandina-veljanovski/).
- [М-р Дарко Пајковски - Асистент - ФИКТ - Битола](https://fikt.uklo.edu.mk/darko-pajkovski/).
- [Акад. Марјан Марковиќ - Дигитални ресурси на македонскиот јазик](https://drmj.manu.edu.mk/).
- [Проф. д-р Георге Гоце Митревски - pelister.org](https://pelister.org/).
- [Ph.D Игор Трајковски - time.mk](https://time.mk/trajkovski/).
- [pesna.org](https://pesna.org/).

<br>

---

### Литература/Референци:
- [Attention is All You need](https://arxiv.org/pdf/1706.03762).
- [What Do Position Embeddings Learn?](https://arxiv.org/pdf/2010.04903)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).
- [Kemal Erdem, (May 2021). "Introduction to Attention Mechanism"](https://erdem.pl/2021/05/introduction-to-attention-mechanism)
- [мал дел од dataset - ]
- [theaisummer.com - Positional Embeddings](https://theaisummer.com/positional-embeddings/)

<br>

---

<br>

## Download

- [Модел Donka v1 -  CC BY-NC 4.0 - Hugging Face ](https://huggingface.co/stefan-n/Donka-v1).
- [Изворен код - Github - Apache]().
- [Google Colab - Тестирање]().

<br>

---

<br>

## Корисна литература за почетници и за љубопитните:

Следниве ресурси се комплетно бесплатни и корисни за сите оние кои сакаат да научат кодирање, машинско учење, линеарна алгебра,веројатност и статистика.

Зошто овие теми, бидејќи со помош на овие теми и вие ќе можите да тренирате и да ги разбирате невронските мрежи и вештачката интелигенција.


- [Линеарна Алгебра - Jim Hefferon - книга кој авторот ја нуди бесплатно](https://hefferon.net/linearalgebra/).
- [Целосен Roadmap за учење](https://rentry.org/machine-learning-roadmap).
- [Научете Python од <b>MIT</b>](https://ocw.mit.edu/courses/6-100l-introduction-to-cs-and-programming-using-python-fall-2022/pages/material-by-lecture/).
- [Научете Python Интеракивно - Курс од <b>Stanford</b>](https://codeinplace.stanford.edu/).
- [Научете Python - Курс од <b>Harvard</b>](https://cs50.harvard.edu/python/2022/weeks/0/).
- [Научете <b>Линеарна Алгебра визуелно</b>, делот со операции за вектори и матрици е многу важен во машинското учење - <b>Immersive Math</b>](https://immersivemath.com/ila/index.html).
- [Визуелно експериментирање со <b> Tensorflow Deep Playground</b>](https://playground.tensorflow.org).
- [<b>3b1b</b> - Невронски Мрежи](https://www.3blue1brown.com/topics/neural-networks).
- [<b>StatQuest</b> - Статистика](https://youtu.be/qBigTkBLU6g?si=oTADX-oND-uB_FVA).


<br>

---

<br>
