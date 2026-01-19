pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker image') {
            steps {
                sh 'docker build -t diabetes-api .'
            }
        }

        stage('Test container') {
            steps {
                sh 'docker run -d --rm -p 8000:8000 --name diabetes-api-test diabetes-api'
                sh "sleep 5"
                sh 'curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d \'{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}\''
            }
        }
    }

    post {
        always {
            sh 'docker stop diabetes-api-test || true'
        }
    }
}

