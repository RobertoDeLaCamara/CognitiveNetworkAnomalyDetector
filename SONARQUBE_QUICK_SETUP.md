# SonarQube Quick Setup Guide for Jenkins

## ✅ What Was Fixed

The Jenkinsfile has been updated to:
1. Use the correct tool name: `'SonarQube Scanner'` (instead of `'SonarScanner'`)
2. Add error handling to gracefully skip SonarQube if not configured
3. Prevent pipeline failures when SonarQube is unavailable

## 🔧 Jenkins Configuration Required

### Step 1: Configure SonarQube Server

**Navigate to:** `Manage Jenkins` → `Configure System` → `SonarQube servers`

1. Click **Add SonarQube**
2. Configure:
   - **Name:** `SonarQube` ⚠️ (must match Jenkinsfile)
   - **Server URL:** `http://localhost:9000` (or your SonarQube server URL)
   - **Server authentication token:**
     1. Generate token in SonarQube: `My Account` → `Security` → `Generate Token`
     2. In Jenkins: Click `Add` → `Jenkins` → `Secret text`
     3. Paste the token
     4. ID: `sonarqube-token`
3. Click **Save**

### Step 2: Configure SonarQube Scanner Tool

**Navigate to:** `Manage Jenkins` → `Global Tool Configuration` → `SonarQube Scanner`

1. Click **Add SonarQube Scanner**
2. Configure:
   - **Name:** `SonarQube Scanner` ⚠️ (must match Jenkinsfile exactly)
   - **Install automatically:** ✅ Check this box
   - **Version:** Select latest version (e.g., SonarQube Scanner 5.0.1.3006)
3. Click **Save**

### Step 3: Install Required Plugin

**Navigate to:** `Manage Jenkins` → `Manage Plugins` → `Available`

1. Search for: `SonarQube Scanner for Jenkins`
2. Install and restart Jenkins if prompted

### Step 4: Configure Webhook (Optional but Recommended)

In SonarQube (for Quality Gate to work):

**Navigate to:** `Administration` → `Configuration` → `Webhooks`

1. Create webhook:
   - **Name:** `Jenkins`
   - **URL:** `http://<JENKINS_URL>/sonarqube-webhook/`
   - Example: `http://192.168.1.62:8080/sonarqube-webhook/`
2. Click **Create**

## 🧪 Testing the Configuration

### Quick Test Pipeline

Run this pipeline to verify SonarQube Scanner is configured:

```groovy
pipeline {
    agent any
    stages {
        stage('Test SonarQube') {
            steps {
                script {
                    def scannerHome = tool name: 'SonarQube Scanner', type: 'hudson.plugins.sonar.SonarRunnerInstallation'
                    echo "SonarQube Scanner found at: ${scannerHome}"
                }
            }
        }
    }
}
```

## 🎯 Expected Names (Must Match!)

| Configuration Location | Setting Name | Required Value |
|------------------------|--------------|----------------|
| Jenkinsfile (line 123) | `tool name:` | `'SonarQube Scanner'` |
| Global Tool Configuration | SonarQube Scanner Name | `SonarQube Scanner` |
| Jenkinsfile (line 128) | `withSonarQubeEnv(...)` | `'SonarQube'` |
| Configure System | SonarQube Server Name | `SonarQube` |

## ⚠️ Common Errors & Solutions

### Error 1: "SonarQube installation not found"
**Solution:** Make sure the name in `Configure System` → `SonarQube servers` is exactly `SonarQube`

### Error 2: "Tool type not found"
**Solution:** Install the plugin: `SonarQube Scanner for Jenkins`

### Error 3: "SonarQube Scanner not configured"
**Solution:** 
1. Go to `Global Tool Configuration`
2. Add `SonarQube Scanner` with name exactly: `SonarQube Scanner`
3. Check "Install automatically"

### Error 4: Quality Gate timeout
**Solution:** Configure webhook in SonarQube pointing to Jenkins

## 📝 What Will Happen Now

With the updated Jenkinsfile:

### ✅ If SonarQube IS configured:
- Pipeline will run SonarQube analysis
- Quality Gate will be checked
- Results will appear in SonarQube dashboard

### ✅ If SonarQube is NOT configured:
- Pipeline will skip SonarQube stages gracefully
- Build will be marked as **UNSTABLE** (not failed)
- Warning messages will be shown
- Other stages (tests, coverage) will still run

## 🚀 Running Your Pipeline

After configuration, commit and push the updated Jenkinsfile:

```bash
cd /home/roberto/cognitive-anomaly-detector
git add Jenkinsfile
git commit -m "fix: Update SonarQube configuration with correct tool name and error handling"
git push
```

Then trigger the Jenkins pipeline and check the console output.

## 📚 Full Documentation

For detailed setup instructions, see [SONARQUBE_JENKINS_SETUP.md](./SONARQUBE_JENKINS_SETUP.md)
