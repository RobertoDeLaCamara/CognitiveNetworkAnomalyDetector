pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '5'))
        timestamps()
        timeout(time: 60, unit: 'MINUTES')
    }

    environment {
        REGISTRY = "192.168.1.86:5000"
        IMAGE_NAME = "cognitive-anomaly-detector"
        NO_PROXY = 'localhost,127.0.0.1,192.168.1.0/24,192.168.1.86,192.168.1.62,192.168.1.45'
        no_proxy = 'localhost,127.0.0.1,192.168.1.0/24,192.168.1.86,192.168.1.62,192.168.1.45'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Image') {
            steps {
                echo 'Building Docker image...'
                sh "docker build -t ${REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER} -t ${REGISTRY}/${IMAGE_NAME}:latest ."
            }
        }

        stage('Code Quality Checks') {
            parallel {
                stage('Lint') {
                    steps {
                        echo 'Running code quality checks...'
                        sh """
                        docker run --rm --user root \
                            ${REGISTRY}/${IMAGE_NAME}:\${BUILD_NUMBER} \
                            sh -c 'pip install --quiet flake8 && flake8 src/ --max-line-length=120 --count --statistics'
                        """
                    }
                }

                stage('Security Checks') {
                    steps {
                        echo 'Checking for security vulnerabilities...'
                        sh """
                        docker run --rm --user root \
                            ${REGISTRY}/${IMAGE_NAME}:\${BUILD_NUMBER} \
                            sh -c 'pip install --quiet pip-audit && pip-audit -r requirements.txt'
                        """
                    }
                }
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running test suite with coverage...'
                script {
                    try {
                        sh """
                        docker run --name test-cad-\${BUILD_NUMBER} \
                            --user root \
                            ${REGISTRY}/${IMAGE_NAME}:\${BUILD_NUMBER} \
                            python -m pytest tests/ -v \
                                --junitxml=test-results.xml \
                                --cov=src \
                                --cov-report=xml:coverage.xml \
                                --cov-report=term-missing \
                                --disable-warnings
                        """
                    } finally {
                        sh "docker cp test-cad-\${BUILD_NUMBER}:/app/test-results.xml \${WORKSPACE}/test-results.xml || true"
                        sh "docker cp test-cad-\${BUILD_NUMBER}:/app/coverage.xml \${WORKSPACE}/coverage.xml || true"
                        sh "docker rm test-cad-\${BUILD_NUMBER} || true"
                    }
                }
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results.xml'
                    archiveArtifacts artifacts: 'coverage.xml', allowEmptyArchive: true, fingerprint: true
                }
            }
        }

        stage('SonarQube Analysis') {
            steps {
                echo 'Running SonarQube analysis...'
                withCredentials([usernamePassword(
                    credentialsId: 'sonarqube-credentials',
                    usernameVariable: 'SONAR_USER',
                    passwordVariable: 'SONAR_PASS'
                )]) {
                    sh """
                        HOST_WORKSPACE=\$(echo \${WORKSPACE} | sed 's|/var/jenkins_home|/home/roberto/jenkins_home|')
                        docker run --rm \
                            -v "\${HOST_WORKSPACE}:/usr/src" \
                            sonarsource/sonar-scanner-cli \
                            -Dsonar.projectKey=cognitive-anomaly-detector \
                            -Dsonar.sources=src \
                            -Dsonar.tests=tests \
                            -Dsonar.python.version=3.11 \
                            -Dsonar.python.coverage.reportPaths=coverage.xml \
                            -Dsonar.host.url=http://192.168.1.86:9000 \
                            -Dsonar.login=\${SONAR_USER} \
                            -Dsonar.password=\${SONAR_PASS} \
                            -Dsonar.scm.disabled=true
                    """
                }
            }
        }

        stage('Push to Registry') {
            steps {
                echo "Pushing image to ${REGISTRY}..."
                sh "docker push ${REGISTRY}/${IMAGE_NAME}:\${BUILD_NUMBER}"
                sh "docker push ${REGISTRY}/${IMAGE_NAME}:latest"
            }
        }
    }

    post {
        always {
            sh 'rm -f test-results.xml coverage.xml || true'
            sh "docker rmi ${REGISTRY}/${IMAGE_NAME}:\${BUILD_NUMBER} || true"
        }
        success {
            echo 'Pipeline succeeded!'
        }
        failure {
            echo 'Pipeline failed.'
        }
    }
}
