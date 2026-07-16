# Milvus向量数据库

## 1 什么是Milvus向量数据库

![image-20250915194718270](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250915194718270.png)

## 2 关键概念

![image-20250915195136221](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250915195136221.png)

![image-20250915195151294](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250915195151294.png)

**注意：1个collection最多支持4个向量Field**

![image-20250915195412281](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250915195412281.png)

![image-20250915195446505](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250915195446505.png)

## 3 为什么选择Milvus

![image-20250915195519824](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250915195519824.png)

## 4 支持哪些索引和度量

![image-20250916110112402](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916110112402.png)

### 1 IVF_FLAT（倒排索引）



![image-20250916110258459](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916110258459.png)

![image-20250916113020323](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916113020323.png)

![image-20250916113119111](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916113119111.png)

![image-20250916113200854](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916113200854.png)

![image-20250916113249663](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916113249663.png)

![image-20250916113314963](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916113314963.png)

![image-20250916113355359](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916113355359.png)

![image-20250916113958806](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916113958806.png)

### 2 IVF_SQ8（标量量化）

![image-20250916111406649](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916111406649.png)

![image-20250916133828444](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916133828444.png)

![image-20250916133840855](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916133840855.png)

![image-20250916133904214](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916133904214.png)

![image-20250916134019762](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916134019762.png)

![image-20250916134118150](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916134118150.png)

![image-20250916134206262](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916134206262.png)

```py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 1. 连接 Milvus
connections.connect("default", host="localhost", port="19530")

# 2. 定义集合 schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)  # 128维向量
]
schema = CollectionSchema(fields, description="IVF_SQ8 example")
collection = Collection("demo_ivf_sq8", schema)

# 3. 插入一些数据
import numpy as np
vectors = np.random.random((1000, 128)).astype("float32").tolist()
ids = list(range(1000))
collection.insert([ids, vectors])

# 4. 创建 IVF_SQ8 索引
index_params = {
    "index_type": "IVF_SQ8",
    "metric_type": "L2",   # 欧式距离，也可以用 IP（内积）
    "params": {"nlist": 64}
}
collection.create_index(field_name="vector", index_params=index_params)

# 5. 加载集合
collection.load()

# 6. 向量检索
query_vectors = np.random.random((1, 128)).astype("float32").tolist()
search_params = {"metric_type": "L2", "params": {"nprobe": 8}}

results = collection.search(
    data=query_vectors,
    anns_field="vector",
    param=search_params,
    limit=5,   # 返回前5个
    output_fields=["id"]
)

# 7. 打印结果
for hits in results:
    for hit in hits:
        print(f"匹配 id={hit.id}, 距离={hit.distance:.4f}")

```

### 3 IVF_PQ（乘积量化）

![image-20250916111457142](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916111457142.png)

![image-20250916134733723](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916134733723.png)

![image-20250916134831764](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916134831764.png)

![image-20250916135843539](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916135843539.png)

![image-20250916140033131](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916140033131.png)

![image-20250916140055935](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916140055935.png)

![image-20250916140109798](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916140109798.png)



![image-20250916135311649](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916135311649.png)

![image-20250916140127556](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916140127556.png)

### 4 相似度度量

![image-20250916163911617](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916163911617.png)

## 5 Milvus数据库操作

![image-20250916165506923](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250916165506923.png)

### 5.1 设置向量数据库[¶](#51)

要创建本地的 Milvus 向量数据库，只需实例化一个`MilvusClient` ，指定一个存储所有数据的文件名，如 "milvus_demo.db"。

我们还需要 在Docker上面启动Milvus。安装好docker。

```py
# -*-coding:utf-8-*-
from pymilvus import MilvusClient, DataType


def operate_db():
    # 如果没有docker，也没有后端启动Milvus服务端的情况下
    # client = MilvusClient(uri='ai_milvus.db')
    # 如果uri为链接地址，代表Milvus属于单机服务，需要开启Milvus后台服务操作
    client = MilvusClient(uri="http://localhost:19530")
    print(f'client-->{client}')
    # 查看库中有多少个databases;
    databases = client.list_databases()
    print(f'databases--》{databases}')
    # # 先判断数据库是否存在，如果不存在，创建，否则直接使用
    # if "milvus_demo" not in databases:
    #     client.create_database(db_name="milvus_demo")
    # else:
    #     client.using_database(db_name='milvus_demo')

    return client

if __name__ == '__main__':
    client = operate_db()
```

结果：

```
client--><pymilvus.milvus_client.milvus_client.MilvusClient object at 0x0000014EC989CEC0>
databases--》['default']
```

![image-20250928194253011](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250928194253011.png)

![image-20250928194349666](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250928194349666.png)

![image-20250928194443711](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250928194443711.png)

![image-20251010162216451](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20251010162216451.png)

### 5.2 Collections操作[¶](#52-collections)

在 Milvus 中，我们需要一个 Collections 来存储向量及其相关元数据。你可以把它想象成传统 SQL 数据库中的表格。创建 Collections 时，可以定义 Schema 和索引参数来配置向量规格，如维度、索引类型和远距离度量。此外，还有一些复杂的概念来优化索引以提高向量搜索性能。

```py
#todo: 2. collection集合的操作
def operate_table():

    # 定义schema
    # 注意：在定义集合 Schema 时，enable_dynamic_field=True 使得您可以插入未定义的字段。
    # 一般动态字段以 JSON 格式存储，通常命名为 $meta。在插入数据时，所有未定义的字段及其值将被保存为键值对。
    # 在定义集合 Schema 时，auto_id=True 可以对主键自动增长id。
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    # 添加字段
    # is_primary=True,意味着当前的字段为主键
    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR, dim=5)
    schema.add_field(field_name='scalar', datatype=DataType.VARCHAR, max_length=256, description="标量字段")

    # 创建一个集合（表）
    client.create_collection(collection_name="demo_v1", schema=schema)

    # 添加索引
    prepare_indexs = client.prepare_index_params()
    prepare_indexs.add_index(field_name='vector', index_type='', metric_type='COSINE', index_name='vector_index')
    client.create_index(collection_name='demo_v1', index_params=prepare_indexs)
    # 查看索引信息
    res = client.list_indexes(collection_name='demo_v1')
    print(f'索引信息--》{res}')
    res1 = client.describe_index(collection_name='demo_v1', index_name="vector_index")
    print(f'索引的详细信息：--》{res1}')
    # 判断集合是否加载:默认：{'state': <LoadState: NotLoad>}
    print(client.get_load_state(collection_name="demo_v1"))
    # {'state': <LoadState: Loaded>}
    client.load_collection(collection_name='demo_v1')
    print(client.get_load_state(collection_name="demo_v1"))
    # # 如果想删除索引，一定要释放集合
    # client.release_collection(collection_name='demo_v1')
    # print(client.get_load_state(collection_name="demo_v1"))
    # # 删除索引
    # client.drop_index(collection_name="demo_v1", index_name="vector_index")

    # 检索标量字段
    index_params1 = client.prepare_index_params()
    prepare_indexs.add_index(field_name='scalar', index_type='', index_name='scalar_index')
    client.create_index(collection_name='demo_v1', index_params=prepare_indexs)
    print(client.list_indexes(collection_name='demo_v1'))
```

### 5.3 Entity实体数据操作[¶](#53-entity)

在 Milvus 中，**实体\**指的是\**Collections\**中共享相同\**Schema** 的数据记录，行中每个字段的数据构成一个实体。因此，同一 Collections 中的实体具有相同的属性（如字段名称、数据类型和其他约束）。

#### 5.3.1数据的增、删、改[¶](#531)

```py

def operate_entity():
    # # todo:1. 创建集合collection
    # 这种方式: collection 只包括两个字段. id 作为主键， vector 作为向量字段，以及自动设置 auto_id、enable_dynamic_field 为 True
    # auto_id 启用此设置可确保主键自动递增。在数据插入期间无需手动提供主键。
    # enable_dynamic_field 启用后，要插入的数据中除 id 和 vector 之外的所有字段都将被视为动态字段。
    # # 这些附加字段作为键值对保存在名为 $meta 的特殊字段中。此功能允许在数据插入期间包含额外的字段。
    # client.create_collection(collection_name='demo_v2', dimension=5, metric_type='IP')

    # # todo:2. 插入数据（也叫实体）
    # data = [
    #     {"id": 0, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354,
    #                          0.9029438446296592], "color": "pink_8682"},
    #     {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501,
    #                          0.838729485096104], "color": "red_7025"},
    #     {"id": 2, "vector": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185,
    #                          0.20785793220625592], "color": "orange_6781"},
    #     {"id": 3, "vector": [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995,
    #                          0.95791889146345], "color": "pink_9298"},
    #     {"id": 4, "vector": [0.4452349528804562, -0.8757026943054742, 0.8220779437047674, 0.46406290649483184,
    #                          0.30337481143159106], "color": "red_4794"},
    #     {"id": 5, "vector": [0.985825131989184, -0.8144651566660419, 0.6299267002202009, 0.1206906911183383,
    #                          -0.1446277761879955], "color": "yellow_4222"},
    #     {"id": 6, "vector": [0.8371977790571115, -0.015764369584852833, -0.31062937026679327, -0.562666951622192,
    #                          -0.8984947637863987], "color": "red_9392"},
    #     {"id": 7, "vector": [-0.33445148015177995, -0.2567135004164067, 0.8987539745369246, 0.9402995886420709,
    #                          0.5378064918413052], "color": "grey_8510"},
    #     {"id": 8, "vector": [0.39524717779832685, 0.4000257286739164, -0.5890507376891594, -0.8650502298996872,
    #                          -0.6140360785406336], "color": "white_9381"},
    #     {"id": 9, "vector": [0.5718280481994695, 0.24070317428066512, -0.3737913482606834, -0.06726932177492717,
    #                          -0.6980531615588608], "color": "purple_4976"}
    # ]
    # res = client.insert(collection_name='demo_v2', data=data)
    # print(res)

    ## todo:2.1 将数据插入到特定分区，可以在插入请求中指定分区名称，如下所示：
    # data = [
    #     {"id": 10, "vector": [-0.5570353903748935, -0.8997887893201304, -0.7123782431855732, -0.6298990746450119,
    #                           0.6699215060604258], "color": "red_1202"},
    #     {"id": 11, "vector": [0.6319019033373907, 0.6821488267878275, 0.8552303045704168, 0.36929791364943054,
    #                           -0.14152860714878068], "color": "blue_4150"},
    #     {"id": 12, "vector": [0.9483947484855766, -0.32294203351925344, 0.9759290319978025, 0.8262982148666174,
    #                           -0.8351194181285713], "color": "orange_4590"},
    #     {"id": 13, "vector": [-0.5449109892498731, 0.043511240563786524, -0.25105249484790804, -0.012030655265886425,
    #                           -0.0010987671273892108], "color": "pink_9619"},
    #     {"id": 14, "vector": [0.6603339372951424, -0.10866551787442225, -0.9435597754324891, 0.8230244263466688,
    #                           -0.7986720938400362], "color": "orange_4863"},
    #     {"id": 15, "vector": [-0.8825129181091456, -0.9204557711667729, -0.935350065513425, 0.5484069690287079,
    #                           0.24448151140671204], "color": "orange_7984"},
    #     {"id": 16, "vector": [0.6285586391568163, 0.5389064528263487, -0.3163366239905099, 0.22036279378888013,
    #                           0.15077052220816167], "color": "blue_9010"},
    #     {"id": 17, "vector": [-0.20151825016059233, -0.905239387635804, 0.6749305353372479, -0.7324272081377843,
    #                           -0.33007998971889263], "color": "blue_4521"},
    #     {"id": 18, "vector": [0.2432286610792349, 0.01785636564206139, -0.651356982731391, -0.35848148851027895,
    #                           -0.7387383128324057], "color": "orange_2529"},
    #     {"id": 19, "vector": [0.055512329053363674, 0.7100266349039421, 0.4956956543575197, 0.24541352586717702,
    #                           0.4209030729923515], "color": "red_9437"}
    # ]
    #
    # # ##  todo:3. 创建分区
    # client.create_partition(collection_name='demo_v2', partition_name='partitionA')
    #
    # # # # # todo: 3.1 分区中插入数据
    # res = client.insert(collection_name='demo_v2', data=data, partition_name='partitionA')
    # print(res)
    # todo:4. 更新插入数据
    # 在 Milvus 中，upsert 操作执行数据级操作，根据集合中是否已存在主键来插入或更新实体。具体来说：
    # 如果集合中已存在该实体的主键，则现有实体将被覆盖。
    # 如果集合中不存在主键，则将插入一个新实体。
    # data = [
    #     {"id": 0, "vector": [-0.619954382375778, 0.4479436794798608, -0.17493894838751745, -0.4248030059917294,
    #                          -0.8648452746018911], "color": "black_9898"},
    #     {"id": 1, "vector": [0.4762662251462588, -0.6942502138717026, -0.4490002642657902, -0.628696575798281,
    #                          0.9660395877041965], "color": "red_7319"},
    #     {"id": 2, "vector": [-0.8864122635045097, 0.9260170474445351, 0.801326976181461, 0.6383943392381306,
    #                          0.7563037341572827],"color": "white_6465"},
    #     {"id": 3, "vector": [0.14594326235891586, -0.3775407299900644, -0.3765479013078812, 0.20612075380355122,
    #                          0.4902678929632145], "color": "orange_7580"},
    #     {"id": 4, "vector": [0.4548498669607359, -0.887610217681605, 0.5655081329910452, 0.19220509387904117,
    #                          0.016513983433433577], "color": "red_3314"},
    #     {"id": 5, "vector": [0.11755001847051827, -0.7295149788999611, 0.2608115847524266, -0.1719167007897875,
    #                          0.7417611743754855], "color": "black_9955"},
    #     {"id": 6, "vector": [0.9363032158314308, 0.030699901477745373, 0.8365910312319647, 0.7823840208444011,
    #                          0.2625222076909237], "color": "yellow_2461"},
    #     {"id": 7, "vector": [0.0754823906014721, -0.6390658668265143, 0.5610517334334937, -0.8986261118798251,
    #                          0.9372056764266794], "color": "white_5015"},
    #     {"id": 8, "vector": [-0.3038434006935904, 0.1279149203380523, 0.503958664270957, -0.2622661156746988,
    #                          0.7407627307791929], "color": "purple_6414"},
    #     {"id": 9, "vector": [-0.7125086947677588, -0.8050968321012257, -0.32608864121785786, 0.3255654958645424,
    #                          0.26227968923834233], "color": "brown_7231"}
    # ]
    #
    # res = client.upsert(collection_name='demo_v2', data=data)
    # print(res)
    # 注意如果分区中不存在更新数据的id，就不会受影响，但是会影响集合里已经存在的相同id的实体
    # res = client.upsert(collection_name='demo_v2', data=data, partition_name="partitionA")
    # todo:5. 删除实体（数据）
    # 按照过滤器删除；如果不指定分区，默认情况下会在整个集合中进行删除
    # res = client.delete(collection_name='demo_v2', filter='id in [12, 5, 6]')
    # print(res)
    # 按照id进行删除；指定分区删除数据  因为 1234不属于A分区 所以删除不了
    res = client.delete(collection_name='demo_v2', ids=[1, 2, 3, 4, 16], partition_name='partitionA')
    print(res)
```

#### 5.3.2 数据的查询[¶](#532)

![image-20250929174225687](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929174225687.png)

```py
from pymilvus import connections, Collection

# 1. 连接 Milvus
connections.connect(alias="default", host="localhost", port="19530")

# 2. 加载集合
collection = Collection("my_collection")
collection.load()

# 3. 定义查询向量  注意必须是2维的
query_vectors = [[0.1, 0.2, 0.3, 0.4]]

# 4. search 操作
results = collection.search(
    data=query_vectors,          # 查询向量
    anns_field="embedding",      # 向量字段名
    param={"metric_type": "L2", "params": {"nprobe": 10}},  
    limit=5,                     # 返回前 5 个最相似向量
    output_fields=["doc_id", "text"]  # 返回额外的字段
)

# 5. 查看结果
for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, Distance: {hit.distance}, Text: {hit.entity.get('text')}")

```

```
def search(
    self,
    data: Union[List, utils.SparseMatrixInputType],
    anns_field: str,
    param: Dict,
    limit: int,
    expr: Optional[str] = None,
    partition_names: Optional[List[str]] = None,
    output_fields: Optional[List[str]] = None,
    timeout: Optional[float] = None,
    round_decimal: int = -1,
    ranker: Optional[Function] = None,
    **kwargs,
):
```

各个字段的含义：

















![image-20250929174718446](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929174718446.png)

![image-20250929174729100](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929174729100.png)

![image-20250929174739609](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929174739609.png)

![image-20250929174849765](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929174849765.png)

**NLP** 一般用 `COSINE` 或 `IP`

![image-20250929175016446](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929175016446.png)

![image-20250929175024638](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929175024638.png)

代码实操：

```py
#todo:4.数据库的查询
def operate_query():
    # todo:1 .单一向量查询
    result1 = client.search(collection_name='demo_v2',
                            data=[[0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104]],
                            limit=2,
                            search_params={"metric_type": "IP"},
                            output_fields=['id', "vector", "color"])
    print(f'result1-->{result1}')
    # todo: 2. 批量向量搜索
    res2 = client.search(collection_name='demo_v2',
                        data=[[0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104],
                              [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345]],
                        limit=2,
                        search_params={"metric_type": "IP"},
                        output_fields=["id", 'vector']) # search_params是在查询时执行距离计算方式，如果定义索引的时候，已经制定了方式可以不写
    print(res2)
    # todo: 3. 分区搜索
    # 要进行分区搜索，只需在搜索请求的 partition_names 中包含目标分区的名称即可。这指定search操作仅考虑指定分区内的向量。
    res3 = client.search(
        collection_name="demo_v2",
        data=[[0.02174828545444263, 0.058611125483182924, 0.6168633415965343, -0.7944160935612321, 0.5554828317581426]],
        limit=5,
        search_params={"metric_type": "IP"},
        partition_names=["partitionA"]  # 这里指定搜索的分区
    )
    print(res3)

    # 使用输出字段进行搜索允许您指定搜索结果中应包含匹配向量的哪些属性或字段。
    res4 = client.search(
        collection_name="demo_v2",
        data=[[0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]],
        limit=5,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=['vector', "color"]  # 返回定义的字段
    )
    print(res4)
    res5 = client.search(
        collection_name="demo_v2",
        data=[[0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]],
        limit=5,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["color"],
        filter='color like "red%"'
    )
    print(res5)
    search_params = {
        "metric_type": "IP",
        "params": {
            "radius": 0.8,  # 搜索圆的半径
            "range_filter": 1  # 范围过滤器，用于过滤出不在搜索圆内的向量。
        }
    }
    # "radius": 0.8,   "range_filter": 1 本质在这里就是一个范围约束[0.8, 1]

    res6 = client.search(
        collection_name="demo_v2",
        data=[[0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]],
        limit=6,  # 返回的搜索结果最大数量
        search_params=search_params,
        output_fields=["color"],
    )
    print(res6)
```

### 5.4 完整代码

```py
# -*-coding:utf-8-*-
from pymilvus import MilvusClient, DataType


def operate_db():
    # 如果没有docker，也没有后端启动Milvus服务端的情况下
    # client = MilvusClient(uri='ai_milvus.db')
    # 如果uri为链接地址，代表Milvus属于单机服务，需要开启Milvus后台服务操作
    client = MilvusClient(uri="http://localhost:19530")
    print(f'client-->{client}')
    # 查看库中有多少个databases;
    databases = client.list_databases()
    print(f'databases--》{databases}')
    # 先判断数据库是否存在，如果不存在，创建，否则直接使用
    # if "milvus_demo" not in databases:
    #     client.create_database(db_name="milvus_demo")
    # else:
    #     client.using_database(db_name='milvus_demo')
    # databases = client.list_databases()
    # print(f'databases--》{databases}')
    return client


#todo: 2. collection集合的操作
def operate_table():

    # 定义schema
    # 注意：在定义集合 Schema 时，enable_dynamic_field=True 使得您可以插入未定义的字段。
    # 一般动态字段以 JSON 格式存储，通常命名为 $meta。在插入数据时，所有未定义的字段及其值将被保存为键值对。
    # 在定义集合 Schema 时，auto_id=True 可以对主键自动增长id。
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    # 添加字段
    # is_primary=True,意味着当前的字段为主键
    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR, dim=5)
    schema.add_field(field_name='scalar', datatype=DataType.VARCHAR, max_length=256, description="标量字段")

    # 创建一个集合（表）
    client.create_collection(collection_name="demo_v1", schema=schema)

    # 添加索引
    prepare_indexs = client.prepare_index_params()
    prepare_indexs.add_index(field_name='vector', index_type='', metric_type='COSINE', index_name='vector_index')
    client.create_index(collection_name='demo_v1', index_params=prepare_indexs)
    # 查看索引信息
    res = client.list_indexes(collection_name='demo_v1')
    print(f'索引信息--》{res}')
    res1 = client.describe_index(collection_name='demo_v1', index_name="vector_index")
    print(f'索引的详细信息：--》{res1}')
    # 判断集合是否加载:默认：{'state': <LoadState: NotLoad>}
    print(client.get_load_state(collection_name="demo_v1"))
    # {'state': <LoadState: Loaded>}
    client.load_collection(collection_name='demo_v1')
    print(client.get_load_state(collection_name="demo_v1"))
    # # 如果想删除索引，一定要释放集合
    # client.release_collection(collection_name='demo_v1')
    # print(client.get_load_state(collection_name="demo_v1"))
    # # 删除索引
    # client.drop_index(collection_name="demo_v1", index_name="vector_index")

    # 检索标量字段
    index_params1 = client.prepare_index_params()
    prepare_indexs.add_index(field_name='scalar', index_type='', index_name='scalar_index')
    client.create_index(collection_name='demo_v1', index_params=prepare_indexs)
    print(client.list_indexes(collection_name='demo_v1'))

def operate_entity():
    # # todo:1. 创建集合collection
    # 这种方式: collection 只包括两个字段. id 作为主键， vector 作为向量字段，以及自动设置 auto_id、enable_dynamic_field 为 True
    # auto_id 启用此设置可确保主键自动递增。在数据插入期间无需手动提供主键。
    # enable_dynamic_field 启用后，要插入的数据中除 id 和 vector 之外的所有字段都将被视为动态字段。
    # # 这些附加字段作为键值对保存在名为 $meta 的特殊字段中。此功能允许在数据插入期间包含额外的字段。
    # client.create_collection(collection_name='demo_v2', dimension=5, metric_type='IP')

    # # todo:2. 插入数据（也叫实体）
    # data = [
    #     {"id": 0, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354,
    #                          0.9029438446296592], "color": "pink_8682"},
    #     {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501,
    #                          0.838729485096104], "color": "red_7025"},
    #     {"id": 2, "vector": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185,
    #                          0.20785793220625592], "color": "orange_6781"},
    #     {"id": 3, "vector": [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995,
    #                          0.95791889146345], "color": "pink_9298"},
    #     {"id": 4, "vector": [0.4452349528804562, -0.8757026943054742, 0.8220779437047674, 0.46406290649483184,
    #                          0.30337481143159106], "color": "red_4794"},
    #     {"id": 5, "vector": [0.985825131989184, -0.8144651566660419, 0.6299267002202009, 0.1206906911183383,
    #                          -0.1446277761879955], "color": "yellow_4222"},
    #     {"id": 6, "vector": [0.8371977790571115, -0.015764369584852833, -0.31062937026679327, -0.562666951622192,
    #                          -0.8984947637863987], "color": "red_9392"},
    #     {"id": 7, "vector": [-0.33445148015177995, -0.2567135004164067, 0.8987539745369246, 0.9402995886420709,
    #                          0.5378064918413052], "color": "grey_8510"},
    #     {"id": 8, "vector": [0.39524717779832685, 0.4000257286739164, -0.5890507376891594, -0.8650502298996872,
    #                          -0.6140360785406336], "color": "white_9381"},
    #     {"id": 9, "vector": [0.5718280481994695, 0.24070317428066512, -0.3737913482606834, -0.06726932177492717,
    #                          -0.6980531615588608], "color": "purple_4976"}
    # ]
    # res = client.insert(collection_name='demo_v2', data=data)
    # print(res)

    ## todo:2.1 将数据插入到特定分区，可以在插入请求中指定分区名称，如下所示：
    # data = [
    #     {"id": 10, "vector": [-0.5570353903748935, -0.8997887893201304, -0.7123782431855732, -0.6298990746450119,
    #                           0.6699215060604258], "color": "red_1202"},
    #     {"id": 11, "vector": [0.6319019033373907, 0.6821488267878275, 0.8552303045704168, 0.36929791364943054,
    #                           -0.14152860714878068], "color": "blue_4150"},
    #     {"id": 12, "vector": [0.9483947484855766, -0.32294203351925344, 0.9759290319978025, 0.8262982148666174,
    #                           -0.8351194181285713], "color": "orange_4590"},
    #     {"id": 13, "vector": [-0.5449109892498731, 0.043511240563786524, -0.25105249484790804, -0.012030655265886425,
    #                           -0.0010987671273892108], "color": "pink_9619"},
    #     {"id": 14, "vector": [0.6603339372951424, -0.10866551787442225, -0.9435597754324891, 0.8230244263466688,
    #                           -0.7986720938400362], "color": "orange_4863"},
    #     {"id": 15, "vector": [-0.8825129181091456, -0.9204557711667729, -0.935350065513425, 0.5484069690287079,
    #                           0.24448151140671204], "color": "orange_7984"},
    #     {"id": 16, "vector": [0.6285586391568163, 0.5389064528263487, -0.3163366239905099, 0.22036279378888013,
    #                           0.15077052220816167], "color": "blue_9010"},
    #     {"id": 17, "vector": [-0.20151825016059233, -0.905239387635804, 0.6749305353372479, -0.7324272081377843,
    #                           -0.33007998971889263], "color": "blue_4521"},
    #     {"id": 18, "vector": [0.2432286610792349, 0.01785636564206139, -0.651356982731391, -0.35848148851027895,
    #                           -0.7387383128324057], "color": "orange_2529"},
    #     {"id": 19, "vector": [0.055512329053363674, 0.7100266349039421, 0.4956956543575197, 0.24541352586717702,
    #                           0.4209030729923515], "color": "red_9437"}
    # ]
    #
    # # ##  todo:3. 创建分区
    # client.create_partition(collection_name='demo_v2', partition_name='partitionA')
    #
    # # # # # todo: 3.1 分区中插入数据
    # res = client.insert(collection_name='demo_v2', data=data, partition_name='partitionA')
    # print(res)
    # todo:4. 更新插入数据
    # 在 Milvus 中，upsert 操作执行数据级操作，根据集合中是否已存在主键来插入或更新实体。具体来说：
    # 如果集合中已存在该实体的主键，则现有实体将被覆盖。
    # 如果集合中不存在主键，则将插入一个新实体。
    # data = [
    #     {"id": 0, "vector": [-0.619954382375778, 0.4479436794798608, -0.17493894838751745, -0.4248030059917294,
    #                          -0.8648452746018911], "color": "black_9898"},
    #     {"id": 1, "vector": [0.4762662251462588, -0.6942502138717026, -0.4490002642657902, -0.628696575798281,
    #                          0.9660395877041965], "color": "red_7319"},
    #     {"id": 2, "vector": [-0.8864122635045097, 0.9260170474445351, 0.801326976181461, 0.6383943392381306,
    #                          0.7563037341572827],"color": "white_6465"},
    #     {"id": 3, "vector": [0.14594326235891586, -0.3775407299900644, -0.3765479013078812, 0.20612075380355122,
    #                          0.4902678929632145], "color": "orange_7580"},
    #     {"id": 4, "vector": [0.4548498669607359, -0.887610217681605, 0.5655081329910452, 0.19220509387904117,
    #                          0.016513983433433577], "color": "red_3314"},
    #     {"id": 5, "vector": [0.11755001847051827, -0.7295149788999611, 0.2608115847524266, -0.1719167007897875,
    #                          0.7417611743754855], "color": "black_9955"},
    #     {"id": 6, "vector": [0.9363032158314308, 0.030699901477745373, 0.8365910312319647, 0.7823840208444011,
    #                          0.2625222076909237], "color": "yellow_2461"},
    #     {"id": 7, "vector": [0.0754823906014721, -0.6390658668265143, 0.5610517334334937, -0.8986261118798251,
    #                          0.9372056764266794], "color": "white_5015"},
    #     {"id": 8, "vector": [-0.3038434006935904, 0.1279149203380523, 0.503958664270957, -0.2622661156746988,
    #                          0.7407627307791929], "color": "purple_6414"},
    #     {"id": 9, "vector": [-0.7125086947677588, -0.8050968321012257, -0.32608864121785786, 0.3255654958645424,
    #                          0.26227968923834233], "color": "brown_7231"}
    # ]
    #
    # res = client.upsert(collection_name='demo_v2', data=data)
    # print(res)
    # 注意如果分区中不存在更新数据的id，就不会受影响，但是会影响集合里已经存在的相同id的实体
    # res = client.upsert(collection_name='demo_v2', data=data, partition_name="partitionA")
    # todo:5. 删除实体（数据）
    # 按照过滤器删除；如果不指定分区，默认情况下会在整个集合中进行删除
    # res = client.delete(collection_name='demo_v2', filter='id in [12, 5, 6]')
    # print(res)
    # 按照id进行删除；指定分区删除数据  因为 1234不属于A分区 所以删除不了
    res = client.delete(collection_name='demo_v2', ids=[1, 2, 3, 4, 16], partition_name='partitionA')
    print(res)


#todo:4.数据库的查询
def operate_query():
    # todo:1 .单一向量查询
    result1 = client.search(collection_name='demo_v2',
                            data=[[0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104]],
                            limit=2,
                            search_params={"metric_type": "IP"},
                            output_fields=['id', "vector", "color"])
    print(f'result1-->{result1}')
    # todo: 2. 批量向量搜索
    res2 = client.search(collection_name='demo_v2',
                        data=[[0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104],
                              [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345]],
                        limit=2,
                        search_params={"metric_type": "IP"},
                        output_fields=["id", 'vector']) # search_params是在查询时执行距离计算方式，如果定义索引的时候，已经制定了方式可以不写
    print(res2)
    # todo: 3. 分区搜索
    # 要进行分区搜索，只需在搜索请求的 partition_names 中包含目标分区的名称即可。这指定search操作仅考虑指定分区内的向量。
    res3 = client.search(
        collection_name="demo_v2",
        data=[[0.02174828545444263, 0.058611125483182924, 0.6168633415965343, -0.7944160935612321, 0.5554828317581426]],
        limit=5,
        search_params={"metric_type": "IP"},
        partition_names=["partitionA"]  # 这里指定搜索的分区
    )
    print(res3)

    # 使用输出字段进行搜索允许您指定搜索结果中应包含匹配向量的哪些属性或字段。
    res4 = client.search(
        collection_name="demo_v2",
        data=[[0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]],
        limit=5,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=['vector', "color"]  # 返回定义的字段
    )
    print(res4)
    res5 = client.search(
        collection_name="demo_v2",
        data=[[0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]],
        limit=5,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["color"],
        filter='color like "red%"'
    )
    print(res5)
    search_params = {
        "metric_type": "IP",
        "params": {
            "radius": 0.8,  # 搜索圆的半径
            "range_filter": 1  # 范围过滤器，用于过滤出不在搜索圆内的向量。
        }
    }
    # "radius": 0.8,   "range_filter": 1 本质在这里就是一个范围约束[0.8, 1]

    res6 = client.search(
        collection_name="demo_v2",
        data=[[0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]],
        limit=6,  # 返回的搜索结果最大数量
        search_params=search_params,
        output_fields=["color"],
    )
    print(res6)


if __name__ == '__main__':
    client = operate_db()
    # operate_table()
    # operate_entity()
    operate_query()
```

### 5.5 复杂查询

- 混合检索：要对两组 ANN 搜索结果进行合并和重新排序，有必要选择适当的重新排序策略。支持两种重排策略：**加权排名策略（WeightedRanker**）和**重排序策略**（**RRFRanker**）。在选择重排策略时，需要考虑的一个问题是，在向量场中是否需要强调一个或多个基本 ANN 搜索。

- **加权排名**：如果您要求结果强调特定的向量场，建议使用该策略。通过 WeightedRanker，您可以为某些向量场分配更高的权重，从而更加强调这些向量场。例如，在多模态搜索中，图片的文字描述可能比图片的颜色更重要。
  - 使用 WeightedRanker 策略时，需要在`WeightedRanker` 函数中输入权重值。混合搜索中的基本 ANN 搜索次数与需要输入的值的次数相对应。输入值的范围应为 [0,1]，数值越接近 1 表示重要性越高。

```py
from pymilvus import WeightedRanker
rerank= WeightedRanker(0.8, 0.3) 
```

![image-20250929192335867](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929192335867.png)

```py
from pymilvus import RRFRanker

ranker = RRFRanker(100)
```

代码解释：

```py
schema = client.create_schema(enable_dynamic_field=False)
# 电影id字段
schema.add_field(field_name='film_id', datatype=DataType.INT64, is_primary=True)
# 电影内容向量字段
schema.add_field(field_name='filmVector', datatype=DataType.FLOAT_VECTOR, dim=5) # 向量字段
# 电影海报向量字段
schema.add_field(field_name="posterVector", datatype=DataType.FLOAT_VECTOR, dim=5) # 向量字段
# 可以分别搜索：
# - 内容相似的电影（使用 filmVector）
# - 海报风格相似的电影（使用 posterVector）
# - 两者结合的混合搜索
```

![image-20250929194010630](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929194010630.png)

![image-20250929194041813](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929194041813.png)

代码解释：

```py
# 创建索引
index_params = client.prepare_index_params()
# field_name='filmVector' 在制定字段上面建立索引  通常是向量字段
# index_type = "IVF_FLAT"  倒排文件索引
index_params.add_index(field_name='filmVector', index_type="IVF_FLAT",
                       metric_type="L2", params={"nlist": 128})
index_params.add_index(field_name='posterVector', index_type="",
                       metric_type="COSINE")
```

![image-20250929193655177](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929193655177.png)

![image-20250929193711868](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929193711868.png)

![image-20250929193734387](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929193734387.png)

![image-20250929193748305](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929193748305.png)

![image-20250929193824737](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929193824737.png)

![image-20250929193844172](C:\Users\13349\AppData\Roaming\Typora\typora-user-images\image-20250929193844172.png)

代码：

```py
from pymilvus import RRFRanker
from pymilvus import MilvusClient, DataType, AnnSearchRequest
import random
from pymilvus import WeightedRanker
rerank= WeightedRanker(0.8, 0.3)

#todo:1. 操作数据库
def operate_db():
    # 如果没有docker，也没有后端启动Milvus服务端的情况下
    # client = MilvusClient(uri='ai_milvus.db')
    # 如果uri为链接地址，代表Milvus属于单机服务，需要开启Milvus后台服务操作
    client = MilvusClient(uri="http://localhost:19530")
    print(f'client-->{client}')
    # 查看库中有多少个databases;
    databases = client.list_databases()
    print(f'databases--》{databases}')
    # # 先判断数据库是否存在，如果不存在，创建，否则直接使用
    # if "milvus_demo" not in databases:
    #     client.create_database(db_name="milvus_demo")
    # else:
    #     client.using_database(db_name='milvus_demo')

    return client

def complex_query():
    # # 定义schema
    schema = client.create_schema(enable_dynamic_field=False)
    # 电影id字段
    schema.add_field(field_name='film_id', datatype=DataType.INT64, is_primary=True)
    # 电影内容向量字段
    schema.add_field(field_name='filmVector', datatype=DataType.FLOAT_VECTOR, dim=5) # 向量字段
    # 电影海报向量字段
    schema.add_field(field_name="posterVector", datatype=DataType.FLOAT_VECTOR, dim=5) # 向量字段
    # 可以分别搜索：
    # - 内容相似的电影（使用 filmVector）
    # - 海报风格相似的电影（使用 posterVector）
    # - 两者结合的混合搜索
    # 定义索引
    index_params = client.prepare_index_params()
    # field_name='filmVector' 在制定字段上面建立索引  通常是向量字段
    # index_type = "IVF_FLAT"  倒排文件索引
    index_params.add_index(field_name='filmVector', index_type="IVF_FLAT",
                           metric_type="L2", params={"nlist": 128})
    index_params.add_index(field_name='posterVector', index_type="",
                           metric_type="COSINE")

    # 创建集合
    # client.create_collection(collection_name='demo_v3', schema=schema, index_params=index_params)

    # 向量库中插入实体
    entities = []
    for  _ in range(1000):
        # 构造实体
        film_id = random.randint(1, 10000)
        film_vector = [random.random() for _ in range(5)]
        poster_vector = [random.random() for _ in range(5)]
        entity = {"film_id": film_id, "filmVector": film_vector, "posterVector": poster_vector}
        entities.append(entity)
    # 插入指定的集合中
    # client.insert(collection_name='demo_v3', data=entities)

    # 多向量查询（注意和批量向量查询不同）
    # 多向量搜索使用 hybrid_search() API 在一次调用中执行多个 ANN 搜索请求。每个 AnnSearchRequest 代表特定矢量场上的单个搜索请求。
    # 示例创建两个 AnnSearchRequest 实例以对两个向量字段执行单独的相似性搜索。
    # 创建多搜索请求 filmVector
    query_filmVector = [[0.8896863042430693, 0.370613100114602, 0.23779315077113428, 0.38227915951132996, 0.5997064603128835]]
    dense_search_params = {"data": query_filmVector,
                           "anns_field": "filmVector",# 该参数值必须与集合模式中使用的值相同。
                           "param": {"metric_type": "L2", "nprobe": 10}, # nprobe代表访问簇的数量
                           "limit": 2}
    request_1 = AnnSearchRequest(**dense_search_params)

    # 创建多搜索请求 posterVector
    query_posterVector = [[0.02550758562349764, 0.006085637357292062, 0.5325251250159071, 0.7676432650114147, 0.5521074424751443]]
    sparse_search_params = {"data": query_posterVector,
                            "anns_field": "posterVector",
                            # 该参数值必须与集合模式中使用的值相同。
                            "param": {"metric_type": "COSINE"},
                            "limit": 2
                            }
    request_2 = AnnSearchRequest(**sparse_search_params)

    reqs = [request_1, request_2]
    ranker = RRFRanker(100)

    res = client.hybrid_search(
        collection_name="demo_v3",
        reqs=reqs,
        ranker=ranker,
        limit=2
    )
    print(res)
    for hits in res:
        print("TopK results:")
        for hit in hits:
            print(hit)



if __name__ == '__main__':
    client = operate_db()
    complex_query()
```



报错信息：

```
Traceback (most recent call last): File "E:\baidu\EduRAG智慧问答系统-配套资料\EduRAG智慧问答系统-配套资料\day01\03-代码\day01\complex_query.py", line 106, in <module> complex_query() ~~~~~~~~~~~~~^^ File "E:\baidu\EduRAG智慧问答系统-配套资料\EduRAG智慧问答系统-配套资料\day01\03-代码\day01\complex_query.py", line 90, in complex_query res = client.hybrid_search( collection_name="demo_v3", ...<2 lines>... limit=2 ) File "C:\Users\13349\AppData\Local\Programs\Python\Python313\Lib\site-packages\pymilvus\milvus_client\milvus_client.py", line 362, in hybrid_search ret.append([hit.to_dict() for hit in hits]) ^^^^ TypeError: 'SequenceIterator' object is not iterable
```

修改源码即可：

```
        ret = []
        print(type(res), res)
        for hits in res:
            print(type(hits), hits)

            ret.append([hit.to_dict() for hit in list(hits)])
```

打印的结果：

```sh
client--><pymilvus.milvus_client.milvus_client.MilvusClient object at 0x00000293A8451160>
databases--》['default', 'milvus_demo']
<class 'pymilvus.client.abstract.SearchResult'> data: ["['id: 4035, distance: 0.009900989942252636, entity: {}', 'id: 4897, distance: 0.009900989942252636, entity: {}']"]
<class 'pymilvus.client.abstract.Hits'> ['id: 4035, distance: 0.009900989942252636, entity: {}', 'id: 4897, distance: 0.009900989942252636, entity: {}']
data: ["[{'id': 4035, 'distance': 0.009900989942252636, 'entity': {}}, {'id': 4897, 'distance': 0.009900989942252636, 'entity': {}}]"] 
TopK results:
{'id': 4035, 'distance': 0.009900989942252636, 'entity': {}}
{'id': 4897, 'distance': 0.009900989942252636, 'entity': {}}
```

