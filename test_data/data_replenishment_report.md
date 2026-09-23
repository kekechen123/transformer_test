# 600k 语料补充报告

- 基础语料：`parallel_wmt_600k_clean_v2.tsv`
- 补充来源：`wmt_zh_en_training_corpus.csv`
- 最终语料：`parallel_wmt_600k_final.tsv`
- 随机种子：`20260923`
- 原始 CSV 数据行：24,752,392
- Reservoir 候选池：30,000
- 实际检查候选：5,462
- 基础句对：594,918
- 新增句对：5,082
- 最终句对：600,000

## 候选淘汰原因

| 原因 | 数量 |
|---|---:|
| 列数错误 | 0 |
| 空文本 | 0 |
| 疑似乱码 | 1 |
| 两侧相同 | 27 |
| 源端疑似非中文 | 8 |
| 目标端疑似非英文 | 0 |
| 英文目标含较多中文 | 8 |
| 长度比例异常 | 2 |
| 字幕时间戳 | 1 |
| 疑似截断残片 | 37 |
| 与基础语料重复 | 295 |
| 候选内部重复 | 1 |

## 方法说明

使用固定随机种子的 reservoir sampling 对整个 CSV 均匀抽样，避免直接取文件开头造成主题偏差。
候选使用与基础语料相同的清洗规则，并额外排除基础语料已有句对和候选内部重复。
原始 CSV 中文列带有预分词空格，补充时合并汉字周围空格，使其与基础语料格式一致。

## 淘汰样例

### 疑似乱码

| CSV 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 1618151 | 谈到时间我从现在到中午佑一系列的面谈我们现在就开始工鬃好吗你能与我们一起共浇午餐吗我很想Philip档恐怕不行。 | and speaking of minutes , I have interviews until noon , so why don " t we get right to work ? can you have lunch with us later ? |

### 两侧相同

| CSV 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 24009364 | Horadiz | Horadiz |
| 20433735 | 2004 - 023C , 2007 - 034C | 2004 - 023C , 2007 - 034C |
| 20488679 | EP | EP |
| 22892659 | 372 . Dong - hyeon Kim | 372 . Dong - hyeon Kim |
| 22792056 | Daimler AG | Daimler AG |

### 源端疑似非中文

| CSV 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 15360465 | Mr . Pouta Jacques Beleyi | Mr. Pouta Jacques Beleyi |
| 10456581 | Mr . Vasily Sidorov * , Mr . Teimouraz Ramishvili * * , | Mr. Vasily Sidorov * , Mr. Teimouraz Ramishvili * * , Mr. Oleg Malguinov * * , |
| 15751831 | C . N . 440.2006 . TREATIES - 9 " Issuance of the corrected version ( Russian authentic text ) of the Protocol " ; | C.N.440.2006.TREATIES - 9 " Issuance of the corrected version ( Russian authentic text ) of the Protocol . " |
| 17533277 | org ; http : / / www . UNDP . org . surf - panama / dgovernance . htlm ; http : / / www . | http : / / www.UNDP.org / surf - panama / dgovernance.htlm ; http : / / www.europeandcis.undp.org / ; |
| 13853795 | CCW / GGE / VIII / WG . 2 / WP . 2 | CCW / GGE / VIII / WG.2 / WP.2 |

### 英文目标含较多中文

| CSV 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 5710502 | 你的年龄我不感兴趣。 | it doesn鈥檛 interest me what you do for a living . |
| 5866790 | 有一个曾经坐在我旁边的同事,在供职40多年之后,不仅拒绝举行欢送会,甚至还不许任何人在他最后一个工作日提及他要离职的事。 | I once sat next to a man who , after more than 40 years鈥 ? service , not only refused the offer of a party , but didn鈥檛 allow anyone to even mention his departure on his last day . |
| 5645732 | ” Jajah的发展,使该公司一位董事亚伊尔?戈德芬格( Yair Goldfinger ) ,想起一项即时讯息服务。 | the company鈥檚 development reminds one Jajah director , Yair Goldfinger , of an instant - messaging service that established a social network in the web鈥檚 younger days . |
| 5955714 | 附件2 :进境水产品输出国家或者地区官方检验检疫证书基本要求一、证书上应当标明:品名(包括学名) 、产地、捕捞区域、加工方式、生产加工企业名称及注册号、出证部门;注明运输工具(船名、航班号、集装箱号等) 、封识号、发货人、收货人、数/重量、生产日期。 | the following information shall be stated in the certificate : product name 锛坕ncluding the formal name锛 ? producing area , fishing area , processing method , names of the production and processing enterprises and the reg |
| 5638383 | 第一百一十一条城市和农村按居民居住地区设立的居民委员会或者村民委员会是基层群众性自治组织。 | article 111.The residents鈥 ? committees and villagers鈥 ? committees established among urban and rural residents on the basis of their place of residence are mass organizations of self - management at the grass - roots le |

### 长度比例异常

| CSV 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 13878421 | 司法 | industry and trade Justice |
| 23996378 | 职位数 | remuneration Tier Number of positions |

### 字幕时间戳

| CSV 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 2886781 | ,谚语)爱屋及乌 | [ 01 : 52.78 ] Love melove my dog |

### 疑似截断残片

| CSV 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 2477479 | : 11 : 10以下记录跟随大卫勇士的首领,就是奋勇帮助他得国,照着耶和华吩咐以色列人的话,与以色列人一同立他作王的 | these also are the chief of the mighty men whom David had who strengthened themselves with him in his kingdom and with all Israel to make him king according to the word of the LORD concerning Israel |
| 2640163 | 字体新华社北京讯— —在美国著名人气社交网络Facebook上,中国总理温家宝拥有44000名支持者,在全球著名政治家中人气排名第六位 | : \| BEIJING , June 3 ( Xinhua ) -- Premier Wen Jiabao has become the sixth most popular politician on the U.S.-based Facebook , a popular social networking site , with more than 44,000 " supporters " |
| 13249799 | *协助书记官长征聘为法院服务的工作人员 | : : to assist the Registrar in recruiting personnel to serve the Court |
| 21661249 | *费用参数:薪金出现改变 | : : cost parameters : change in salary scale |
| 16266905 | *其他代表团则认为,应将"区域代表性"视为负有区域责任的区域席位。 | : : other delegations expressed the view that the term " regional representation " should be understood as the regional seat leading to regional accountability . |

### 与基础语料重复

| CSV 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 9404099 | 乍得 | Chad |
| 22173720 | 协调 | coordination |
| 11673635 | 清洁发展机制登记册要求 | clean development mechanism registry requirements |
| 15859261 | 共计 | total |
| 23124003 | 增进人人享有文化权利和尊重文化多样性 | promotion of the enjoyment of the cultural rights of everyone and respect for cultural diversity |

### 候选内部重复

| CSV 行号 | 中文源端 | 英文目标端 |
|---:|---|---|
| 19478293 | 一.导言 | I. Introduction |

## 新增句对样例

| 中文源端 | 英文目标端 |
|---|---|
| 这是我们权力范围内的事. | this matter comes within our jurisdiction . |
| 人们有时死于可治之症. | people sometimes die of treatable conditions . |
| 然而,由于必须要用尽国内的维权手段后才能援引《任择议定书》 ,受害者必须认识到《公约》赋予的权利,以便在国内法庭进行援引。 | however , since local remedies had to be exhausted before the Optional Protocol could be invoked , victims must be aware of their rights under the Convention so that they could invoke it in the national courts . |
| 如果你认真思考片刻,就知道这是件很荒谬的事. | if you think about it for a minute , It 'sounds kind of ridiculous . |
| 这种方法包括:指定一个区域伙伴或一个与该区域关系密切的国家担任报告编写指导,或请秘书长委派一名官员负责这项工作。 | these could include the appointment of a regional partner or another State with close relations with the region to act as a mentor in the preparation of reports , or a request to the Secretary - General to give this job  |
| 学校有很多教室,目前在容纳村里儿童之外还有剩余。 | the school had more rooms than were currently needed to accommodate the village 's children . |
| 最近的两次会议和研讨会是1995年3月28日和29日在坦桑尼亚阿鲁沙举行的民主国家的立宪制度和法律制度会议及1996年6月18日和19日在坦桑尼亚阿鲁沙举办的立宪制度研讨会。 | the latest such Conferences and Seminars have been a Conference on " Constitutionalism and the Legal System in a Democracy , " 28 - 29 March 1995 , Arusha , Tanzania , and Seminar on " Constitutionalism , " Arusha , Tanz |
| 战略遗产计划拟议在所有会议室安装当前通用技术设备,替换局部配电设备和电源插座以增大容量,并更新和扩大电话通信系统的容量; | under the strategic heritage plan , it is proposed that current technology in all of the conference rooms be installed , that local distribution equipment and outlets be replaced in order to enlarge capacity and that the |
| 数控布带缠绕机是复合材料成型工艺中使用的关键设备之一。 | we developed an NC ( numerical control ) tape winding machine for composite forming of aerospace components . |
| 阀瓣打开后,压杆在弹簧作用下旋转一定角度支住阀瓣,使其不能关闭。 | when the disc is opened , with the springs at work , the strut rotates to an angle so as to prop up the disc and keep it from closing . |
| 14 .根据大会第60 / 91号决议第7段,秘书长通过2006年3月7日的一项说明,向裁军审议委员会转递了裁军谈判会议的年度报告,以及大会第六十届会议有关裁军事项的所有正式记录( A / CN . 10 / 203 ) 。 | 14 . pursuant to paragraph 7 of General Assembly resolution 60 / 91 , the Secretary - General , by a note dated 7 March 2006 , transmitted to the Disarmament Commission the annual report of the Conference on Disarmament  |
| 塞纳河隐没在各座桥下,而各座桥又隐没在房屋下面。 | the Seine was hidden by bridges , the bridges by houses . |
| 打一炮就跑的事情我不干。 | hit it and quit it is not my thing . |
| 他就在外面的某个地方,是不是? | he really is out there , isn 't he ? |
| 根据1971年《精神药物公约》第二条第一款和第三款,中国政府通知秘书长其建议将氯胺酮添入《 1971年公约》附表一。 | pursuant to article 2 , paragraphs 1 and 3 , of the Convention on Psychotropic Substances of 1971 , the Government of China notified the Secretary - General of its recommendation that ketamine should be added to Schedule |
| 96 .表7显示的是各国在旧轮胎和报废轮胎方面适用的各种管理体系。 | table 7 shows the respective management systems used by countries for used and scrap tyres . |
| C .工作方法7 - 10 4 | C. Methods of work 7 - 10 5 |
| 媒体,包括塞拉利昂记者协会和编辑行会等重要机构在内,必须负起更大责任,作出公正无偏、符合事实的报道。 | the media , including important institutions like the Sierra Leone Association of Journalists and Editors ' Guild , must assume greater responsibility for unbiased and factual reporting . |
| 根据提交人,因而缔约国不能坚持这两个部门是独立的。 | according to her , the State party can therefore not maintain that those sections are independent . |
| 17 .佛得角共和国总统佩德罗·维罗纳·罗德里格斯·皮雷斯先生阁下 | 17 . his Excellency Mr. Pedro Verona Rodrigues Pires , President of the Republic of Cape Verde |
