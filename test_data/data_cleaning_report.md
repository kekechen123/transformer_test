# 中英平行语料清洗报告

- 输入文件：`parallel_wmt_600k_clean.tsv`
- 输出文件：`parallel_wmt_600k_clean_v2.tsv`
- 输入句对：600,000
- 保留句对：594,918
- 删除句对：5,082（0.85%）
- 发生文本规范化：406,754

## 删除原因

| 原因 | 数量 | 占输入比例 |
|---|---:|---:|
| 列数错误 | 0 | 0.000% |
| 空文本 | 0 | 0.000% |
| 疑似乱码 | 264 | 0.044% |
| 两侧相同 | 2,320 | 0.387% |
| 源端疑似非中文 | 1,233 | 0.205% |
| 目标端疑似非英文 | 10 | 0.002% |
| 英文目标含较多中文 | 866 | 0.144% |
| 长度比例异常 | 153 | 0.026% |
| 归一化后重复 | 236 | 0.039% |

## 保留数据长度统计

| 字符数 | P50 | P90 | P95 | P99 | 最大值 |
|---|---:|---:|---:|---:|---:|
| 中文源端 | 34 | 76 | 92 | 122 | 445 |
| 英文目标端 | 117 | 266 | 318 | 413 | 565 |

## 规则说明

文本规范化包括 HTML 实体还原、`@-@` 连字符还原、Unicode NFKC 和空白合并。
过滤规则有意保持保守：明确乱码、同文/重复、明显语言方向错误，以及字符长度比大于 12 或小于 0.35 的长文本错配。
语言判断采用字符启发式，不替代专业语言识别或句对语义评分；训练前建议抽样复核下列样例。

## 删除样例

### 疑似乱码

| 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 232 | 命运宠爱勇敢者 , 但是抛弃胆小者 . | fortune favors the bold , but  abandons  the timid . |
| 673 | 自己幸福自己创。 | 宸辨墍涓嶆  锛屽嬁澶变簬浜恒 € ? |
| 745 |   圣经里面记述撒玛利亚城里的人陷入了严重的饥荒,食物极其缺乏。 | the Bible says the people of Samaria suffered a great famine , which means there is no food . |
| 3453 |  如果穿着网球鞋去参加宴会,别人会认为你很古怪。 | you 'll be considered eccentric if you go to the banquet in your tennis shoes . |
| 3716 | “是啊,也许我真的爱一个贫苦的姑娘, ”尼古拉自言自语地说, “怎么,我要为财产而牺牲爱情和荣誉吗? | 鈥淵es , perhaps I really do love a poor girl , 鈥 ? Nikolay said to himself ; 鈥渨hat , am I to sacrifice my feeling and my honour for fortune ? |

### 两侧相同

| 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 96 | Algeria | Algeria |
| 424 | Kraljevic | Kraljevic |
| 494 | Elda Moreno | Elda Moreno |
| 638 | 7 ( a ) | 7 ( a ) |
| 1387 | 10 . Saleh Al - Rohal | 10 . Saleh Al - Rohal |

### 源端疑似非中文

| 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 716 | Original : English | original : English |
| 731 | Ms . Samira Sajarova | Ms. Samira Sajarova |
| 2242 | UNEP / OzL . Conv . 7 / 2 - | UNEP / OzL.Conv.7 / 2- |
| 3440 | Global Water Partnership | global Water Partnership |
| 3514 | UNEP / FAO / RC / COP . 7 / 18 | UNEP / FAO / RC / COP.7 / 18 |

### 目标端疑似非英文

| 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 23933 | " 自己的好事别去提,别人的恩惠要铭记。 | 徒有良好的愿望而不去努力实现 , 后悔莫及 . |
| 80253 | 护短是加倍的错误。 | 两个错误 , 加不出一个正确 . |
| 98932 | 世上唯有贫穷可以不劳而获。 | 瀹佸彲楗胯倸瀛愶紝鍒囪帿鍘诲 € 熷 € ? |
| 162522 | . 听起来很不错。 | 相对于我的身高来说 , 体重太重了 ! |
| 229700 | 亡羊补牢,为时未晚。 | 未做好的活 , 需要重新做 . |

### 英文目标含较多中文

| 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 3697 | 假若你要先到那里就请为我保留一小块空间靠近我失去的两位亲人 - 那最小的“睡袍”对我会合适和仅仅一点点“花冠” - 你知道当我们回家我们不在意穿着我很高兴我不信它因它会停止我的呼吸 - 而我愿意多看上一眼这样一个稀奇古怪的尘世! | if you should get there first Save just a little space for me Close to the two I lost The smallest 鈥淩obe鈥 ? will fit me And just a bit of 鈥淐rown鈥 ? for you know we do not mind our dress When we are going home I鈥檓 glad I  |
| 6611 | 德银证券 ( Deutsche Bank Securities ) 首席经济学家、美联储前官员彼得 ? 胡珀 ( Peter Hooper ) 表示: “人们确实认识到,民主党在获得这些委员会的领导权后,希望更慎重地审视美联储的‘双重使命 ' 。 | 鈥淥ne does get the sense that the Democrat leadership on these committees is going to want to weight a little more carefully the dual mandate , 鈥 ? says Peter Hooper , chief economist at Deutsche Bank Securities and a for |
| 8785 | 第五十二条中华人民共和国公民有维护国家统一和全国各民族团结的义务。 | article 52.It is the duty of citizens of the People鈥檚 Republic of China to safeguard the unity of the country and the unity of all its nationalities . |
| 9605 | 泽纳维 ( Meles Zenawi ) 最近表示,该国政府“非常希望”为投资方提供数十万公顷的农业用地。 | Meles Zenawi , prime minister of Ethiopia , said recently its government was 鈥渧ery eager鈥 ? to provide hundreds of thousands of hectares of agricultural land for investment . |
| 10497 | 中国 6 月份外汇储备增长创两年多以来的最低水平,令人们猜测中国政府举措已导致投机性资本流入中国的速度减缓。 | China鈥檚 foreign exchange reserves grew at their slowest pace in more than two years in June , prompting questions over whether the flow of speculative capital into China is slowing as a result of government efforts . |

### 长度比例异常

| 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 6211 | 件数 | amount of compensation claimed |
| 8703 | 发展社团 | Eritrean Communities Mutual Assistance and Cultural Development in USA |
| 17569 | 建议 | implementation of previous recommendations of the Board of Auditors |
| 17925 | 无庸细述。 | this needn 't be related in detail . there is no need to go into details . |
| 20287 | 1 待印发。 | ( VII ) Implementation of the Declaration on the Granting of Independence to Colonial Countries and Peoples |

### 归一化后重复

| 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 2204 | 联合国 | United Nations |
| 4553 | 一 . 导言 | I. Introduction |
| 6169 | 段次页次 | paragraphs Page |
| 6589 | 二 . 文职人员 | II . civilian personnel |
| 11806 | 联合国 | United NATIONS |

