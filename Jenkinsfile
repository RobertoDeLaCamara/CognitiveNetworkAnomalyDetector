pipeline {
    agent any
    
    environment {
        // Python version
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
                script {
                    // Instala Python3 si no existe
                    sh '''
                        if ! command -v python3 &> /dev/null; then
                    if command -v apt-get &> /dev/null; then
                        apt-get update && apt-get install -y python3 python3-venv python3-pip
                    elif command -v apk &> /dev/null; then
                        apk add python3 py3-pip py3-virtualenv
                    fi
                fi
                
                [ ! -d venv ] && python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
            '''
                }
    }
}

        
        stage('Code Quality Checks') {
            parallel {
                stage('Lint') {
                    steps {
                        echo 'Running code quality checks...'
                        sh '''
                            . $VENV_DIR/bin/activate
                            # Optional: Add linting if you have flake8 or pylint
                            # pip install flake8
                            # flake8 src/ --max-line-length=120 --exclude=venv
                            echo "Linting skipped - add linter to requirements if needed"
                        '''
                    }
                }
                
                stage('Security Checks') {
                    steps {
                        echo 'Checking for security vulnerabilities...'
                        sh '''
                            . $VENV_DIR/bin/activate
                            # Optional: Add safety check
                            # pip install safety
                            # safety check -r requirements.txt --json
                            echo "Security checks skipped - add safety to requirements if needed"
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
                        # Don't fail the build, just warn
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
            // Clean up test results directory for next run
            sh 'mkdir -p test-results'
        }
        
        success {
            echo '✅ Build succeeded! All tests passed.'
            // Optional: Send notification
        }
        
        failure {
            echo '❌ Build failed! Check test results.'
            // Optional: Send notification
        }
        
        unstable {
            echo '⚠️ Build unstable! Some tests may have failed.'
        }
    }
}
