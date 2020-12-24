# 自然语言处理大作业

本项目为2020年度秋季学期自然语言处理大作业——机器翻译的代码

### 代码依赖

本代码使用python 3.8.3开发，代码依赖torch，jieba，matplotlib和nltk运行。

可使用以下命令在conda env中安装依赖环境

```
conda create -n ml python=3.8
pip install -r requirements.txt
```

### 代码说明

#### `prepare_data.py`

`prepare_data.py`文件可以利用原始文件生成`input_lang`，`output_lang`，`pairs`三个文件，其中前两个文件是源语言和目标语言的实例，主要实现序列与单词之间的互相转换；pairs保存了满足筛选条件的句子对，详细筛选条件见实验报告。

由于原始数据有10M大小，因此将文件进行一次预处理后，将`prepare_data.py`生成文件利用`pickle`转换为二进制文件，二进制文件已上传至北航云盘【[下载链接](https://bhpan.buaa.edu.cn:443/link/8844CA23421C1DA09CF61156AE571740)】（无需解压，直接放在./data文件夹下即可）

#### `config.py`

`config.py`文件是模型配置文件，其中保存了用来训练模型的各个参数，以及输入模型，图像和预处理文件的地址。

#### `utils.py`

`utils.py`主要实现了一些数据加工（将原始数据转换为模型可以读入的数据），计时函数（记录现在运行程序的时间）和图像绘制函数（损失函数）

#### `base_model.py`

`base_model.py`文件为基础的encoder和decoder模型，参考Pytorch官网教程的实现方法，将单例执行的模型改为支持batch执行，在本次作业中主要给注意力模型做对比。

#### `attention_model.py`

`attention_model.py`实现了基于注意力机制的decoder，能够支持三种注意力的计算方法：dot, general, concat

#### `train.py`

`train.py`为模型训练文件，其中含有预处理文件读入，模型训练，模型BLEU值计算，模型保存等函数。

### 训练模型

注：当前代码中使用的batch_size为100。若出现显存溢出的情况，需要在`config.py`进行修改

```
python train.py
```

运行完成后，BLEU得分最高的模型将保存在./model下，通过baseline和attention机制训练的模型会分别存放。

### 演示Demo

注：演示需要训练好的翻译模型，可先运行代码进行训练，或在北航云盘【[下载链接](https://bhpan.buaa.edu.cn:443/link/FD99747C89C2AE843EB7CD8AB60A6A58)】中下载训练好的模型（无需解压，直接放在./model文件夹下即可）

```
python demo.py
```

执行将自动生成一百个样例，样例格式如下所示：

```
中文句子1> 她 说 她 对 你 说 “ 抱歉
标准答案1> she said to tell you'sorry
英文句子1> she said she said she said you
中文句子2> 在 我 轻 嗅 爱 的 花朵 时
标准答案2> bees buzz inside my heart
英文句子2> and i found the the the of
中文句子3> 好 吧 ， 男人 ， 你 上 吧
标准答案3> all right , butch , you're up
英文句子3> okay , you on on on your
中文句子4> 把 枪 给 我 开枪 。 射 她
标准答案4> give me the gun. shoot her
英文句子4> put me to the the the her
中文句子5> 我 就 知道 。 我 是 对 的
标准答案5> i knew it. i was right
英文句子5> i knew i was i'm
```





