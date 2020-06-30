之前的Neural Machine Translation基本上都是基于word单词作为基本单位的，但是其缺点是不能很好的解决out-of-vocabulary即单词不在词汇库里的情况，且对于单词的一些词法上的修饰(morphology)处理的也不是很好。一个自然的想法就是能够利用比word更基本的组成来建立模型，以更好的解决这些问题。

# Character-Level Model

一种思路是将字符作为基本单元，建立Character-level model，但是由于基本单元换为字符后，相较于单词，其输入的序列更长了，使得数据更稀疏且长程的依赖关系更难学习，训练速度也会降低。[Fully Character-Level Neural Machine Translation without Explicit Segmentation](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1610.03017)中利用了多层的convolution, pooling与highway layer来解决这一问题，其中encoder的结构如下图所示：

![img](CS224N%20Lecture%2012%20-%20Information%20from%20parts%20of%20words%20Subword%20Models.assets/v2-e37c464f9d3c1101c8f0f9c372005047_1440w.jpg)

输入的字符先被映射到character embedding。然后与窗口大小不同的卷积核进行卷积操作再将输出联结起来，例如上图中有三种窗口大小分别为3，4，5的卷积核，相当于学习了基于字符的trigram, 4-grams, 5-grams。然后对卷积的输出进行max pooling操作，相当于选择最显著的特征产生segment embedding。由此我们从最基础的输入的character embedding得到了系统中认为语言学上有意义的segment embedding。然后将这些特征经过Highway Network(有些类似于Residual network，方便深层网络中信息的流通，不过加入了一些控制信息流量的gate）和双向的GRU，这样得到最终的encoder output。之后decoder再利用Attention机制以及character level GRU进行decode。

实验结果显示，基于字符的模型能更好的处理OOV的问题，而且对于多语言场景，能更好的学习各语言间通用的词素。

# Byte Pair Encoding与SentencePiece

基本单元介于字符与单词之间的模型称作Subword Model。那么Subword如何选择呢？一种方法是Byte Pair Encoding,简称BPE。 BPE最早是一种压缩算法，基本思路是把经常出现的byte pair用一个新的byte来代替，例如假设('A', ’B‘）经常顺序出现，则用一个新的标志'AB'来代替它们。

给定了文本库，我们的初始词汇库仅包含所有的单个的字符，然后不断的将出现频率最高的n-gram pair作为新的ngram加入到词汇库中，直到词汇库的大小达到我们所设定的某个目标为止。

例如，假设我们的文本库中出现的单词及其出现次数为 {'l o w': 5, 'l o w e r': 2, 'n e w e s t': 6, 'w i d e s t': 3}，我们的初始词汇库为{ 'l', 'o', 'w', 'e', 'r', 'n', 'w', 's', 't', 'i', 'd'}，出现频率最高的ngram pair是('e','s') 9次，所以我们将'es'作为新的词汇加入到词汇库中，由于'es'作为一个整体出现在词汇库中，这时文本库可表示为 {'l o w': 5, 'l o w e r': 2, 'n e w es t': 6, 'w i d es t': 3}，这时出现频率最高的ngram pair是('es','t') 9次，将'est'加入到词汇库中，文本库更新为{'l o w': 5, 'l o w e r': 2, 'n e w est': 6, 'w i d est': 3}，新的出现频率最高的ngram pair是('l','o')7次，将'lo'加入到词汇库中，文本库更新为{'lo w': 5, 'lo w e r': 2, 'n e w est': 6, 'w i d est': 3}。以此类推，直到词汇库大小达到我们所设定的目标。这个例子中词汇量较小，对于词汇量很大的实际情况，我们就可以通过BPE逐步建造一个较小的基于subword unit的词汇库来表示所有的词汇。

谷歌的NMT模型用了BPE的变种，称作wordpiece model，BPE中利用了n-gram count来更新词汇库，而wordpiece model中则用了一种贪心算法来最大化语言模型概率，即选取新的n-gram时都是选择使得perplexity减少最多的ngram。进一步的，sentencepiece model将词间的空白也当成一种标记，可以直接处理sentence，而不需要将其pre-tokenize成单词。

# Hybrid Model

还有一种思路是在大多数情况下我们还是采用word level模型，而只在遇到OOV的情况才采用character level模型。

其结构如下图所示，大部分还是依赖于比较高效的word level模型，但遇到例子中的"cute"这样的OOV词汇，我们就需要建立一个character level的表示，decode时遇到<unk>这个表示OOV的特殊标记时，就需要character level的decode，训练过程是end2end的，不过损失函数是word部分与character level部分损失函数的加权叠加。

![img](CS224N%20Lecture%2012%20-%20Information%20from%20parts%20of%20words%20Subword%20Models.assets/v2-9447728f421604f7997ceb52b3ed0870_1440w.jpg)

# FastText

fasttext是facebook开源的一个词向量与文本分类工具，在2016年开源，典型应用场景是“带监督的文本分类问题”， fastText结合了自然语言处理和机器学习中最成功的理念。这些包括了使用词袋以及n-gram袋表征语句，还有使用子词(subword)信息，并通过隐藏表征在类别间共享信息。另外采用了一个softmax层级(利用了类别不均衡分布的优势)来加速运算过程。