pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                // Checkout your code from version control
                git 'https://github.com/girinath18/text_to_image/blob/main/main.py'
            }
        }
        
        stage('Setup Environment') {
            steps {
                // Set up the Python environment
                sh 'python3 -m venv venv'
                sh 'source venv/bin/activate'
                sh 'pip install -r requirements.txt'  // You should have a requirements.txt file with your dependencies
            }
        }
        
        stage('Run Python Script') {
            steps {
                // Run your Python script
                sh 'python your_script.py'
            }
        }
        
        stage('Publish Artifacts') {
            steps {
                // Archive the output files
                archiveArtifacts artifacts: 'output.jpg', onlyIfSuccessful: true
            }
        }
    }
    
    post {
        success {
            // If the build is successful, do something
            echo 'Build successful!'
        }
        failure {
            // If the build fails, do something
            echo 'Build failed!'
        }
    }
}
