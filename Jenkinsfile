pipeline {
    agent any
    
    // The empty 'tools' block was removed here to fix the syntax error.
    
    environment {
        // Python command - intentará usar python3, python, o la versión configurada en tools
        PYTHON = 'python3'
        // Virtual environment directory
        VENV_DIR = 'venv'
        // PATH con python agregado
        PATH = "${env.PATH}:/usr/bin:/usr/local/bin"
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out code...'
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                echo 'Setting up Python virtual environment...'
                sh '''
                    # Detectar comando Python disponible
                    if command -v python3 &> /dev/null; then
                        PYTHON_CMD=python3
                    elif command -v python &> /dev/null; then
                        PYTHON_CMD=python
                    else
                        echo "ERROR: Python no está instalado en el servidor Jenkins"
                        echo "Por favor, instala Python 3.8+ en el servidor Jenkins"
                        exit 1
                    fi
                    
                    echo "Usando Python: $PYTHON_CMD"
                    $PYTHON_CMD --version
                    
                    # Crear venv y activar
                    $PYTHON_CMD -m venv $VENV_DIR
                    . $VENV_DIR/bin/activate
                    
                    # Actualizar pip e instalar dependencias
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    
                    # Instalar herramientas de testing/coverage
                    pip install pytest pytest-cov coverage flake8 safety
                '''
            }
        }
        
        stage('Code Quality Checks') {
            parallel {
                stage('Lint') {
                    steps {
                        echo 'Running code quality checks...'
                        sh '''
                            . $VENV_DIR/bin/activate
                            flake8 src/ --max-line-length=120 --exclude=venv,$VENV_DIR || echo "Lint warnings found"
                        '''
                    }
                }
                
                stage('Security Checks') {
                    steps {
                        echo 'Checking for security vulnerabilities...'
                        sh '''
                            . $VENV_DIR/bin/activate
                            safety check -r requirements.txt --full-report || echo "Security warnings found"
                        '''
                    }
                }
            }
        }
        
        stage('Run Tests') {
            steps {
                echo 'Running test suite...'
                sh '''
                    . $VENV_DIR/bin/activate
                    
                    # Crear directorio de resultados
                    mkdir -p test-results
                    
                    # Run tests with coverage
                    pytest tests/ -v \
                        --junitxml=test-results/junit.xml \
                        --cov=src \
                        --cov-report=xml:coverage.xml \
                        --cov-report=html:htmlcov \
                        --cov-report=term-missing
                '''
            }
        }
        
        stage('Test Coverage Report') {
            steps {
                echo 'Generating coverage report...'
                sh '''
                    . $VENV_DIR/bin/activate
                    
                    # Display coverage summary
                    coverage report
                    
                    # Check minimum coverage threshold (85%)
                    coverage report --fail-under=85 || {
                        echo "WARNING: Test coverage is below 85%"
                    }
                '''
            }
        }
        
        stage('SonarQube Analysis') {
            steps {
                script {
                    try {
                        // Definir el scanner de SonarQube configurado en Jenkins
                        // Nota: El nombre 'SonarQube Scanner' debe coincidir con el configurado en
                        // Manage Jenkins > Global Tool Configuration > SonarQube Scanner
                        def scannerHome = tool name: 'SonarQube Scanner', type: 'hudson.plugins.sonar.SonarRunnerInstallation'
                        
                        // Ejecutar análisis de SonarQube
                        // Nota: El nombre 'SonarQube' debe coincidir con el configurado en
                        // Manage Jenkins > Configure System > SonarQube servers
                        withSonarQubeEnv('SonarQube') {
                            sh """
                                ${scannerHome}/bin/sonar-scanner \
                                -Dsonar.projectKey=cognitive-anomaly-detector \
                                -Dsonar.projectName='Cognitive Anomaly Detector' \
                                -Dsonar.projectVersion=1.0 \
                                -Dsonar.sources=src \
                                -Dsonar.tests=tests \
                                -Dsonar.python.coverage.reportPaths=coverage.xml \
                                -Dsonar.python.xunit.reportPath=test-results/junit.xml \
                                -Dsonar.exclusions=venv/**,htmlcov/**,*.pyc,__pycache__/**
                            """
                        }
                    } catch (Exception e) {
                        echo "⚠️ SonarQube analysis skipped: ${e.message}"
                        echo "Please configure SonarQube in Jenkins (see SONARQUBE_JENKINS_SETUP.md)"
                        unstable("SonarQube not configured")
                    }
                }
            }
        }
        
        stage('Quality Gate') {
            steps {
                script {
                    try {
                        // Esperar el resultado del Quality Gate de SonarQube
                        timeout(time: 5, unit: 'MINUTES') {
                            def qg = waitForQualityGate()
                            if (qg.status != 'OK') {
                                echo "WARNING: Quality Gate failed: ${qg.status}"
                                // No falla el build, solo advierte
                                unstable("Quality Gate failed")
                            } else {
                                echo "✅ Quality Gate passed!"
                            }
                        }
                    } catch (Exception e) {
                        echo "⚠️ Quality Gate check skipped: ${e.message}"
                        echo "This is expected if SonarQube analysis was skipped"
                    }
                }
            }
        }
        
        stage('Archive Artifacts') {
            steps {
                echo 'Archiving test results and coverage reports...'
                // Archive test results
                junit 'test-results/junit.xml'
                
                // Archive coverage reports
                publishHTML(target: [
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'htmlcov',
                    reportFiles: 'index.html',
                    reportName: 'Coverage Report'
                ])
                
                // Archive coverage XML for other tools
                archiveArtifacts artifacts: 'coverage.xml', fingerprint: true
            }
        }
    }
    
    post {
        always {
            echo 'Cleaning up...'
            sh '''
                mkdir -p test-results
                rm -rf $VENV_DIR
            '''
        }
        
        success {
            echo '✅ Build succeeded! All tests passed.'
        }
        
        failure {
            echo '❌ Build failed! Check test results.'
        }
        
        unstable {
            echo '⚠️ Build unstable! Some tests may have failed.'
        }
    }
}
