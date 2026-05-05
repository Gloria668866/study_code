# CI/CD 持续集成持续交付

## 一、CI/CD 概念与价值

### 1.1 什么是 CI/CD

CI/CD 是持续集成（Continuous Integration）和持续交付/部署（Continuous Delivery/Deployment）的统称，是实现 DevOps 的核心实践。

```
开发提交代码 → 持续集成(CI) → 持续交付(CD) → 生产环境
           ↓              ↓              ↓
      代码审查      自动化构建     自动化部署
      单元测试      自动化测试     蓝绿/金丝雀
      静态扫描      制品产出      监控回滚
```

### 1.2 CI/CD 核心价值

| 维度 | 传统方式 | CI/CD 方式 |
|------|----------|-----------|
| 集成频率 | 数周/数月一次 | 每天多次 |
| 构建方式 | 手动 | 自动化 |
| 测试覆盖 | 低 | 高（自动化测试流水线） |
| 发布周期 | 数周/数月 | 分钟/小时级 |
| 问题发现 | 推迟到大集成阶段 | 提交时即反馈 |
| 回滚能力 | 困难耗时 | 一键回滚 |

### 1.3 CI/CD Pipeline 典型阶段

```
触发 → 代码检出 → 编译/构建 → 单元测试 → 静态分析 →
制品打包 → 部署到测试环境 → 集成测试 → 部署到预发布环境 → 
冒烟测试 → 部署到生产环境 → 监控验证
```

## 二、Jenkins Pipeline 编写

### 2.1 Jenkins 简介

Jenkins 是最广泛使用的开源 CI/CD 工具，通过 Pipeline（流水线）实现构建、测试、部署全流程自动化。

### 2.2 声明式 Pipeline 示例

```groovy
pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'registry.example.com'
        APP_NAME = 'myapp'
        APP_VERSION = "${BUILD_NUMBER}"
        SONAR_HOST_URL = 'http://sonarqube:9000'
    }
    
    parameters {
        choice(name: 'DEPLOY_ENV', choices: ['dev', 'staging', 'prod'], description: '部署环境')
        booleanParam(name: 'RUN_INTEGRATION_TESTS', defaultValue: true, description: '是否运行集成测试')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    // 获取git提交信息
                    env.GIT_COMMIT_ID = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    env.GIT_BRANCH = sh(script: 'git rev-parse --abbrev-ref HEAD', returnStdout: true).trim()
                }
            }
        }
        
        stage('Code Scan') {
            parallel {
                stage('Lint Check') {
                    steps {
                        sh 'pip install flake8'
                        sh 'flake8 src/ --max-line-length=120'
                    }
                }
                stage('Security Scan') {
                    steps {
                        sh 'pip install bandit'
                        sh 'bandit -r src/ -f html -o bandit_report.html'
                    }
                }
            }
        }
        
        stage('Build') {
            steps {
                sh '''
                    docker build -t ${DOCKER_REGISTRY}/${APP_NAME}:${APP_VERSION} .
                    docker tag ${DOCKER_REGISTRY}/${APP_NAME}:${APP_VERSION} ${DOCKER_REGISTRY}/${APP_NAME}:latest
                '''
            }
        }
        
        stage('Test') {
            parallel {
                stage('Unit Test') {
                    steps {
                        sh 'pip install pytest pytest-cov'
                        sh 'pytest tests/unit --cov=src --cov-report=xml --junitxml=test-results.xml'
                    }
                    post {
                        always {
                            junit 'test-results.xml'
                        }
                    }
                }
                stage('Integration Test') {
                    when {
                        expression { params.RUN_INTEGRATION_TESTS }
                    }
                    steps {
                        sh 'pytest tests/integration/ --junitxml=integration-results.xml'
                    }
                }
            }
        }
        
        stage('Quality Gate') {
            steps {
                withSonarQubeEnv('SonarQube') {
                    sh 'sonar-scanner -Dsonar.projectKey=${APP_NAME} -Dsonar.sources=src/'
                }
                timeout(time: 10, unit: 'MINUTES') {
                    waitForQualityGate abortPipeline: true
                }
            }
        }
        
        stage('Push Image') {
            steps {
                sh '''
                    docker push ${DOCKER_REGISTRY}/${APP_NAME}:${APP_VERSION}
                    docker push ${DOCKER_REGISTRY}/${APP_NAME}:latest
                '''
            }
        }
        
        stage('Deploy') {
            when {
                expression { params.DEPLOY_ENV == 'prod' }
            }
            input {
                message '确认部署到生产环境？'
                ok '确认部署'
            }
            steps {
                script {
                    // Helm部署
                    sh """
                        helm upgrade --install ${APP_NAME} ./helm-chart \\
                            --set image.tag=${APP_VERSION} \\
                            --set environment=${DEPLOY_ENV} \\
                            --namespace ${DEPLOY_ENV}
                    """
                }
            }
        }
    }
    
    post {
        success {
            script {
                sh 'echo "Pipeline 成功! 版本: ${APP_VERSION} 已部署到 ${DEPLOY_ENV}"'
            }
        }
        failure {
            script {
                // 发送告警
                slackSend(
                    color: 'danger',
                    message: "Pipeline 失败: ${env.JOB_NAME} #${env.BUILD_NUMBER} (<${env.BUILD_URL}|详情>)"
                )
            }
        }
    }
}
```

## 三、GitLab CI 配置

GitLab CI/CD 通过 `.gitlab-ci.yml` 文件定义流水线，与代码库紧密集成。

```yaml
# .gitlab-ci.yml
image: docker:24

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: ""
  APP_NAME: myapp

stages:
  - lint
  - test
  - build
  - deploy

# 缓存pip依赖
.pip_cache: &pip_cache
  cache:
    key: ${CI_COMMIT_REF_SLUG}
    paths:
      - .cache/pip
      - venv/

lint:flake8:
  stage: lint
  image: python:3.11-slim
  before_script:
    - pip install flake8
  script:
    - flake8 src/ --max-line-length=120 --statistics
  only:
    - merge_requests
    - master

unit-tests:
  stage: test
  image: python:3.11-slim
  <<: *pip_cache
  before_script:
    - pip install pytest pytest-cov
  script:
    - pytest tests/ --cov=src --cov-report=term-missing --junitxml=report.xml
  artifacts:
    reports:
      junit: report.xml
    paths:
      - htmlcov/
  coverage: '/TOTAL.*\s+(\d+%)$/'

build-image:
  stage: build
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
  only:
    - master
    - tags

deploy-staging:
  stage: deploy
  image: dtzar/helm-kubectl:latest
  before_script:
    - kubectl config use-context staging
  script:
    - helm upgrade --install $APP_NAME ./chart \
        --set image.tag=$CI_COMMIT_SHORT_SHA \
        --namespace staging
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - master

deploy-production:
  stage: deploy
  image: dtzar/helm-kubectl:latest
  before_script:
    - kubectl config use-context production
  script:
    - helm upgrade --install $APP_NAME ./chart \
        --set image.tag=$CI_COMMIT_SHORT_SHA \
        --namespace production
  environment:
    name: production
    url: https://prod.example.com
  when: manual
  only:
    - tags
```

## 四、自动化构建与部署流程

### 4.1 制品管理

制品（Artifact）是构建过程产生的可部署文件，应使用专门的制品仓库管理：

```bash
# Docker 镜像 → Harbor / ECR
# Java JAR/WAR → Nexus / JFrog Artifactory
# 前端静态资源 → 阿里云 OSS / AWS S3
# Helm Chart → Chart Museum / OCI Registry
```

### 4.2 部署策略实现

**滚动更新（Kubernetes）**：
```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxUnavailable: 25%
    maxSurge: 25%
```

**金丝雀发布（Ingress + 权重路由）**：
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: app-canary
spec:
  hosts:
  - api.example.com
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: app-service
        subset: v2
  - route:
    - destination:
        host: app-service
        subset: v1
      weight: 90
    - destination:
        host: app-service
        subset: v2
      weight: 10
```

## 五、CI/CD 最佳实践

1. **一切纳入版本控制**：代码、配置、Pipeline 定义全部用 Git 管理
2. **快速失败原则**：运行快的测试放在前面（lint、unit test），慢的放在后面
3. **环境一致性**：开发、测试、生产环境尽可能使用相同的基础镜像和配置
4. **制品不可变**：每个构建产生唯一版本的制品，部署到不同环境的是同一个制品
5. **自动回滚**：部署失败时自动触发回滚，减少 MTTR（平均恢复时间）
6. **安全扫描左移**：在 CI 阶段就进行 SAST（静态安全测试）和依赖漏洞扫描
7. **Pipeline 即代码**：使用 Jenkinsfile / .gitlab-ci.yml / GitHub Actions 管理流水线

CI/CD 的核心不是工具本身，而是通过自动化的流水线缩短反馈周期、提高交付质量和速度的文化与实践。
