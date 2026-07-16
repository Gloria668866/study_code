CREATE DATABASE IF NOT EXISTS voice_analysis DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE voice_analysis;

CREATE TABLE IF NOT EXISTS tb_interview_recording_analysis (
    id INT NOT NULL AUTO_INCREMENT COMMENT '唯一标识ID',
    name VARCHAR(100) NOT NULL COMMENT '姓名',
    interview_time DATETIME NOT NULL COMMENT '面试时间',
    company_name VARCHAR(255) NOT NULL COMMENT '公司名',
    subject VARCHAR(255) NULL COMMENT '面试学科',
    recording_url VARCHAR(255) NOT NULL COMMENT '录音地址',
    processing_status SMALLINT DEFAULT 0 COMMENT '处理进度（0：未处理，1：正在处理，2：处理完成）',
    processing_tips TEXT NULL COMMENT '处理提示',
    overall_comments TEXT NULL COMMENT '整体点评',
    interview_score FLOAT NULL COMMENT '面试评分',
    strengths TEXT NULL COMMENT '优势点',
    weaknesses TEXT NULL COMMENT '不足点',
    improvement_suggestions TEXT NULL COMMENT '改进建议',
    interview_text TEXT NULL COMMENT '面试文本',
    markdown_text TEXT NULL COMMENT '面试评价格式生成',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '修改时间',
    PRIMARY KEY (id),
    INDEX ind_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS tb_interview_recording_analysis_detail (
    id INT NOT NULL AUTO_INCREMENT COMMENT '唯一标识ID',
    interview_record_analysis_id VARCHAR(32) NOT NULL COMMENT '面试记录分析ID',
    interview_question TEXT NULL COMMENT '面试问题',
    interviewee_answer TEXT NULL COMMENT '面试者回答',
    reference_answer TEXT NULL COMMENT '参考答案',
    point_analysis TEXT NULL COMMENT '考点分析',
    answer_thoughts TEXT NULL COMMENT '答题思路',
    answer_evaluation TEXT NULL COMMENT '回答评价',
    answer_score FLOAT NULL COMMENT '回答评分',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '修改时间',
    PRIMARY KEY (id),
    INDEX ind_interview_record_analysis_id (interview_record_analysis_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS tb_user (
    user_id VARCHAR(32) NOT NULL COMMENT '用户ID',
    user_name VARCHAR(100) NOT NULL COMMENT '用户名',
    password VARCHAR(255) NOT NULL COMMENT '密码',
    PRIMARY KEY (user_id),
    UNIQUE KEY uk_user_name (user_name),
    INDEX ind_user_id (user_id),
    INDEX ind_user_name (user_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
