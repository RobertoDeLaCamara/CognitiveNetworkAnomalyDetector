pipeline {
    agent {
        docker {
            image 'python:3.11-slim'
            //label 'docker'  // Nombre de tu Docker installation en Global Tool Config
            args '-u root --network=host -v /var/run/docker.sock:/var/run/docker.sock:/var/run/docker.sock'
        }
    }
    
    environment {
        // Python version (ya incluido en la imagen Docker)
        PYTHON = 'python3'
        // Virtual environment directory
        VENV_DIR = 'venv'
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
                    # Crear venv y activar
                    python3 -m venv $VENV_DIR
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
