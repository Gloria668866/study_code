# MyBatis持久层框架原理与使用详解

## 一、MyBatis概述

MyBatis是一款优秀的Java持久层框架，它消除了几乎所有的JDBC代码和手动参数设置，以及结果集的检索。MyBatis可以通过简单的XML或注解来配置和映射原生类型、接口和Java POJO到数据库记录。

**MyBatis的核心特点**：
- **SQL与代码分离**：SQL写在XML文件中，便于管理和优化
- **动态SQL**：强大的动态SQL构建能力，解决复杂查询痛点
- **灵活的映射**：支持自动映射和自定义映射规则
- **插件机制**：通过拦截器扩展框架功能
- **缓存支持**：内置一级缓存和二级缓存
- **与Spring无缝集成**：通过MyBatis-Spring模块

**与JPA/Hibernate的区别**：
- MyBatis是半自动ORM，需要手写SQL（适合复杂查询场景）
- JPA/Hibernate是全自动ORM，SQL自动生成（适合标准CRUD场景）
- MyBatis对SQL有完全控制权，便于性能调优

## 二、MyBatis核心组件

### 2.1 SqlSessionFactory

`SqlSessionFactory`是MyBatis的核心对象，用于创建`SqlSession`。整个应用生命周期中应该只有一个实例。

```java
// 通过XML配置创建SqlSessionFactory
String resource = "mybatis-config.xml";
InputStream inputStream = Resources.getResourceAsStream(resource);
SqlSessionFactory sqlSessionFactory = new SqlSessionFactoryBuilder().build(inputStream);

// Spring Boot中自动配置
// 只需配置数据源，MyBatis自动创建SqlSessionFactory
```

### 2.2 SqlSession

`SqlSession`是执行SQL、获取映射器、管理事务的核心接口。每个线程应有自己的SqlSession实例，它是非线程安全的。

```java
// 通过SqlSession执行SQL
try (SqlSession session = sqlSessionFactory.openSession()) {
    // 方式1：直接执行SQL
    User user = session.selectOne("com.example.mapper.UserMapper.selectById", 1L);
    
    // 方式2：通过Mapper接口（推荐）
    UserMapper mapper = session.getMapper(UserMapper.class);
    User user2 = mapper.selectById(1L);
    
    session.commit();  // 需要手动提交
}
```

### 2.3 Mapper接口

Mapper接口是MyBatis最常用的开发方式，将Java接口与SQL映射绑定。

```java
// Mapper接口定义
@Mapper  // 或使用@MapperScan在配置类上统一扫描
public interface UserMapper {
    
    User selectById(Long id);
    
    List<User> selectByCondition(@Param("name") String name, 
                                  @Param("status") Integer status);
    
    int insert(User user);
    
    int update(User user);
    
    int deleteById(Long id);
}
```

## 三、MyBatis XML映射文件详解

### 3.1 基础映射配置

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
    "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
    
<mapper namespace="com.example.mapper.UserMapper">
    
    <!-- 结果映射 -->
    <resultMap id="BaseResultMap" type="com.example.entity.User">
        <id column="id" property="id" jdbcType="BIGINT"/>
        <result column="user_name" property="userName" jdbcType="VARCHAR"/>
        <result column="email" property="email" jdbcType="VARCHAR"/>
        <result column="status" property="status" jdbcType="TINYINT"/>
        <result column="create_time" property="createTime" jdbcType="TIMESTAMP"/>
        <result column="update_time" property="updateTime" jdbcType="TIMESTAMP"/>
    </resultMap>
    
    <!-- SQL片段（可复用） -->
    <sql id="BaseColumns">
        id, user_name, email, status, create_time, update_time
    </sql>
    
    <!-- 基础查询 -->
    <select id="selectById" resultMap="BaseResultMap">
        SELECT <include refid="BaseColumns"/>
        FROM t_user
        WHERE id = #{id}
    </select>
    
    <!-- 插入（自动生成主键） -->
    <insert id="insert" parameterType="com.example.entity.User"
            useGeneratedKeys="true" keyProperty="id" keyColumn="id">
        INSERT INTO t_user (user_name, email, status, create_time, update_time)
        VALUES (#{userName}, #{email}, #{status}, NOW(), NOW())
    </insert>
    
    <!-- 更新 -->
    <update id="update" parameterType="com.example.entity.User">
        UPDATE t_user
        <set>
            <if test="userName != null">user_name = #{userName},</if>
            <if test="email != null">email = #{email},</if>
            <if test="status != null">status = #{status},</if>
            update_time = NOW()
        </set>
        WHERE id = #{id}
    </update>
    
    <!-- 删除 -->
    <delete id="deleteById">
        DELETE FROM t_user WHERE id = #{id}
    </delete>
</mapper>
```

### 3.2 动态SQL

动态SQL是MyBatis最强大的特性之一，解决了根据不同条件拼接SQL的痛点。

```xml
<!-- 条件查询：if + where -->
<select id="selectByCondition" resultMap="BaseResultMap">
    SELECT <include refid="BaseColumns"/>
    FROM t_user
    <where>
        <if test="name != null and name != ''">
            AND user_name LIKE CONCAT('%', #{name}, '%')
        </if>
        <if test="status != null">
            AND status = #{status}
        </if>
        <if test="email != null and email != ''">
            AND email = #{email}
        </if>
    </where>
    ORDER BY create_time DESC
</select>

<!-- 批量插入：foreach -->
<insert id="batchInsert">
    INSERT INTO t_user (user_name, email, status, create_time)
    VALUES
    <foreach collection="list" item="user" separator=",">
        (#{user.userName}, #{user.email}, #{user.status}, NOW())
    </foreach>
</insert>

<!-- 批量更新（使用case when） -->
<update id="batchUpdate">
    UPDATE t_user
    <trim prefix="SET" suffixOverrides=",">
        <trim prefix="status = CASE" suffix="END,">
            <foreach collection="list" item="user">
                WHEN id = #{user.id} THEN #{user.status}
            </foreach>
        </trim>
    </trim>
    WHERE id IN
    <foreach collection="list" item="user" open="(" separator="," close=")">
        #{user.id}
    </foreach>
</update>

<!-- 条件选择：choose/when/otherwise（类似switch） -->
<select id="selectByPriority" resultMap="BaseResultMap">
    SELECT * FROM t_user
    <where>
        <choose>
            <when test="id != null">
                id = #{id}
            </when>
            <when test="email != null">
                email = #{email}
            </when>
            <when test="name != null">
                user_name = #{name}
            </when>
            <otherwise>
                status = 1
            </otherwise>
        </choose>
    </where>
</select>

<!-- set标签：自动处理逗号 -->
<update id="updateSelective">
    UPDATE t_user
    <set>
        <if test="userName != null">user_name = #{userName},</if>
        <if test="email != null">email = #{email},</if>
        <if test="status != null">status = #{status},</if>
    </set>
    WHERE id = #{id}
</update>

<!-- trim标签：更灵活的前缀后缀处理 -->
<select id="selectByTrim" resultMap="BaseResultMap">
    SELECT * FROM t_user
    <trim prefix="WHERE" prefixOverrides="AND | OR">
        <if test="name != null">AND user_name = #{name}</if>
        <if test="status != null">AND status = #{status}</if>
    </trim>
</select>

<!-- bind标签：创建变量 -->
<select id="selectByLike" resultMap="BaseResultMap">
    <bind name="pattern" value="'%' + keyword + '%'"/>
    SELECT * FROM t_user
    WHERE user_name LIKE #{pattern}
</select>
```

### 3.3 关联查询（一对一、一对多、多对多）

```xml
<!-- ========== 一对一关联 ========== -->
<!-- User包含一个Department对象 -->
<resultMap id="UserWithDeptMap" type="com.example.entity.User" extends="BaseResultMap">
    <association property="department" 
                 javaType="com.example.entity.Department"
                 column="dept_id"
                 select="com.example.mapper.DepartmentMapper.selectById"/>
</resultMap>

<!-- 或者使用联合查询（JOIN方式，避免N+1问题） -->
<resultMap id="UserWithDeptJoinMap" type="com.example.entity.User">
    <id column="id" property="id"/>
    <result column="user_name" property="userName"/>
    <!-- association使用association标签 -->
    <association property="department" javaType="com.example.entity.Department">
        <id column="dept_id" property="id"/>
        <result column="dept_name" property="name"/>
    </association>
</resultMap>

<select id="selectUserWithDept" resultMap="UserWithDeptJoinMap">
    SELECT u.*, d.id as dept_id, d.name as dept_name
    FROM t_user u
    LEFT JOIN t_department d ON u.dept_id = d.id
    WHERE u.id = #{id}
</select>

<!-- ========== 一对多关联 ========== -->
<!-- Department包含多个User -->
<resultMap id="DeptWithUsersMap" type="com.example.entity.Department">
    <id column="id" property="id"/>
    <result column="name" property="name"/>
    <collection property="users" ofType="com.example.entity.User">
        <id column="user_id" property="id"/>
        <result column="user_name" property="userName"/>
        <result column="email" property="email"/>
    </collection>
</resultMap>

<select id="selectDeptWithUsers" resultMap="DeptWithUsersMap">
    SELECT d.id, d.name, u.id as user_id, u.user_name, u.email
    FROM t_department d
    LEFT JOIN t_user u ON d.id = u.dept_id
    WHERE d.id = #{id}
</select>

<!-- ========== 多对多关联 ========== -->
<!-- 用户和角色多对多 -->
<resultMap id="UserWithRolesMap" type="com.example.entity.User" extends="BaseResultMap">
    <collection property="roles" ofType="com.example.entity.Role"
                select="com.example.mapper.RoleMapper.selectByUserId"
                column="id"/>
</resultMap>
```

**N+1查询问题及解决方案**：
- 分步查询（association的select属性）会导致N+1问题
- 优先使用JOIN查询（一条SQL获取所有数据）
- 或使用延迟加载（lazyLoadingEnabled=true）+ 批量查询（MyBatis-Plus）

## 四、MyBatis缓存机制

### 4.1 一级缓存（SqlSession级别）

一级缓存是SqlSession级别的缓存，默认开启，无法关闭。

```java
// 一级缓存演示
try (SqlSession session = sqlSessionFactory.openSession()) {
    UserMapper mapper = session.getMapper(UserMapper.class);
    
    User user1 = mapper.selectById(1L);  // 查询数据库，放入一级缓存
    User user2 = mapper.selectById(1L);  // 命中缓存，不查数据库
    System.out.println(user1 == user2);  // true（同一对象）
}
```

**一级缓存失效的情况**：
- 不同的SqlSession
- 同一个SqlSession但查询条件不同
- 两次查询之间执行了增删改操作（清空缓存）
- 手动调用了`sqlSession.clearCache()`

### 4.2 二级缓存（Mapper/Namespace级别）

二级缓存是跨SqlSession的，需要手动配置开启。

```xml
<!-- mybatis-config.xml -->
<settings>
    <setting name="cacheEnabled" value="true"/>  <!-- 全局开启二级缓存 -->
</settings>

<!-- UserMapper.xml -->
<mapper namespace="com.example.mapper.UserMapper">
    <!-- 开启二级缓存 -->
    <cache 
        eviction="LRU"            <!-- 淘汰策略：LRU/FIFO/SOFT/WEAK -->
        flushInterval="60000"     <!-- 刷新间隔（毫秒） -->
        size="512"                <!-- 缓存对象数量上限 -->
        readOnly="true"/>         <!-- 只读缓存（性能更好，但返回的是同一实例） -->
    
    <!-- 单个查询控制缓存 -->
    <select id="selectById" resultMap="BaseResultMap" useCache="true">
        SELECT * FROM t_user WHERE id = #{id}
    </select>
    
    <!-- 清除二级缓存 -->
    <update id="update" flushCache="true">
        UPDATE t_user SET ... WHERE id = #{id}
    </update>
</mapper>
```

**二级缓存注意事项**：
- 查询结果所关联的实体对象必须实现Serializable接口
- 在分布式环境下，需要使用Redis等分布式缓存替代MyBatis二级缓存
- MyBatis的二级缓存与Spring事务集成可能存在问题，推荐使用Spring Cache + Redis

## 五、MyBatis插件机制

MyBatis插件通过拦截器实现，可以拦截Executor、ParameterHandler、ResultSetHandler、StatementHandler。

```java
// 自定义分页拦截器示例
@Intercepts({
    @Signature(
        type = Executor.class,
        method = "query",
        args = {MappedStatement.class, Object.class, RowBounds.class, ResultHandler.class}
    )
})
public class PaginationInterceptor implements Interceptor {
    
    @Override
    public Object intercept(Invocation invocation) throws Throwable {
        Object[] args = invocation.getArgs();
        MappedStatement ms = (MappedStatement) args[0];
        Object parameter = args[1];
        RowBounds rowBounds = (RowBounds) args[2];
        
        // 如果不需要分页，直接执行原查询
        if (rowBounds == RowBounds.DEFAULT) {
            return invocation.proceed();
        }
        
        // 获取原始SQL
        BoundSql boundSql = ms.getBoundSql(parameter);
        String originalSql = boundSql.getSql();
        
        // 拼接分页SQL
        String pageSql = originalSql + " LIMIT " + rowBounds.getOffset() + ", " + rowBounds.getLimit();
        
        // 创建新的MappedStatement（略）
        // 执行分页查询和count查询
        
        return invocation.proceed();
    }
    
    @Override
    public Object plugin(Object target) {
        return Plugin.wrap(target, this);
    }
}
```

## 六、MyBatis-Plus增强

MyBatis-Plus是在MyBatis基础上的增强工具，提供了更多便捷功能。

```java
// BaseMapper：内置通用CRUD操作
public interface UserMapper extends BaseMapper<User> {
    // 无需手写简单CRUD
    // 内置方法：insert, deleteById, updateById, selectById, selectList等
}

// 条件构造器（LambdaWrapper，类型安全）
@Service
public class UserServiceImpl implements UserService {
    @Autowired
    private UserMapper userMapper;
    
    public List<User> searchUsers(String keyword, Integer status) {
        LambdaQueryWrapper<User> wrapper = new LambdaQueryWrapper<>();
        wrapper.like(StringUtils.hasText(keyword), User::getUserName, keyword)
               .eq(status != null, User::getStatus, status)
               .ge(User::getCreateTime, LocalDate.now().minusDays(7))
               .orderByDesc(User::getCreateTime);
        
        return userMapper.selectList(wrapper);
    }
    
    // 分页查询
    public IPage<User> pageUsers(int pageNum, int pageSize) {
        Page<User> page = new Page<>(pageNum, pageSize);
        LambdaQueryWrapper<User> wrapper = new LambdaQueryWrapper<>();
        wrapper.orderByDesc(User::getCreateTime);
        
        return userMapper.selectPage(page, wrapper);
    }
}

// 逻辑删除（自动处理）
@Entity
@TableLogic(value = "0", delval = "1")
public class User {
    private Integer deleted;  // 0未删除，1已删除
}
// 删除操作自动变为：UPDATE t_user SET deleted=1 WHERE id=?

// 乐观锁
@Entity
public class Product {
    @Version
    private Integer version;
}
// 更新时会自动增加version检查
```

## 七、性能优化实践

### 7.1 SQL优化原则

1. **避免SELECT ***：只查询需要的列，减少网络传输和内存占用
2. **合理使用批量操作**：使用foreach批量插入（注意SQL长度限制，建议每批500-1000条）
3. **使用JOIN替代子查询**：大多数场景JOIN性能优于子查询
4. **合理建立索引**：根据查询条件为常用字段建立索引
5. **避免大事务**：长事务会造成锁竞争和连接占用
6. **分页优化**：大数据量分页使用游标方式（基于条件过滤+Limit加索引覆盖）

### 7.2 MyBatis特定优化

```xml
<!-- 大数据量查询使用流式查询（fetchSize） -->
<select id="selectAllByCursor" resultMap="BaseResultMap" fetchSize="1000">
    SELECT * FROM t_user
</select>

<!-- 使用#{ }而非${ }防止SQL注入（除非动态表名/列名） -->
<!-- 安全：参数化查询 -->
<select id="safeQuery">SELECT * FROM t_user WHERE id = #{id}</select>
<!-- 危险：字符串拼接，可能SQL注入 -->
<select id="unsafeQuery">SELECT * FROM ${tableName} WHERE id = #{id}</select>
```

## 八、总结

MyBatis作为Java持久层的主流方案，以其灵活性和对SQL的完全控制而广受欢迎。掌握XML映射配置、动态SQL、缓存机制和插件开发，能够应对各种复杂的数据访问需求。实际项目中，推荐结合MyBatis-Plus使用，以减少模板代码编写。同时，合理使用JOIN查询避免N+1问题，配置合适的缓存策略，是提升MyBatis应用性能的关键。
