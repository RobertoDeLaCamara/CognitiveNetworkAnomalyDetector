# Configuración de SonarQube en Jenkins

Este documento describe los pasos necesarios para configurar SonarQube en Jenkins sin usar Docker.

## Requisitos Previos

1. **Jenkins** instalado y ejecutándose
2. **SonarQube Server** instalado y ejecutándose (puede estar en el mismo servidor o en uno diferente)
3. **Python 3** instalado en el servidor Jenkins ⚠️ **CRÍTICO**
4. **SonarQube Scanner** instalado en el servidor Jenkins

## Paso 0: Instalar Python en el Servidor Jenkins (OBLIGATORIO)

> [!IMPORTANT]
> Si Jenkins está ejecutándose en un contenedor Docker, necesitas instalar Python dentro del contenedor.

### Opción A: Script Automático (Recomendado)

Usa el script proporcionado `install-python-jenkins.sh`:

```bash
# Si Jenkins está en Docker, accede al contenedor
docker exec -it -u root <nombre-contenedor-jenkins> bash

# Dentro del contenedor, ejecuta:
cd /var/jenkins_home/workspace/cognitive-anomaly-detector
bash install-python-jenkins.sh
```

### Opción B: Instalación Manual

```bash
# Acceder al contenedor de Jenkins (si aplica)
docker exec -it -u root <nombre-contenedor-jenkins> bash

# Actualizar repositorios
apt-get update

# Instalar Python 3 y herramientas necesarias
apt-get install -y python3 python3-pip python3-venv python3-dev build-essential

# Verificar instalación
python3 --version
pip3 --version
```

### Opción C: Usar una Imagen Custom de Jenkins

Crea un `Dockerfile` personalizado basado en Jenkins:

```dockerfile
FROM jenkins/jenkins:lts

USER root

# Instalar Python 3
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

USER jenkins
```

Luego construye y ejecuta:

```bash
docker build -t jenkins-python .
docker run -d -p 8080:8080 -p 50000:50000 jenkins-python
```

### Verificar Python en Jenkins

Para verificar que Python está disponible, ejecuta un pipeline de prueba:

```groovy
pipeline {
    agent any
    stages {
        stage('Test Python') {
            steps {
                sh 'python3 --version'
                sh 'pip3 --version'
            }
        }
    }
}
```

## Paso 1: Instalar SonarQube Scanner en el Servidor Jenkins

### Opción A: Instalación Manual

```bash
# Descargar SonarQube Scanner
cd /opt
wget https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-5.0.1.3006-linux.zip

# Descomprimir
unzip sonar-scanner-cli-5.0.1.3006-linux.zip
mv sonar-scanner-5.0.1.3006-linux sonar-scanner

# Dar permisos
chmod +x sonar-scanner/bin/sonar-scanner
```

### Opción B: Instalación Automática desde Jenkins

Jenkins puede descargar e instalar automáticamente el scanner (configuración en el paso 3).

## Paso 2: Instalar Plugins de Jenkins

Instala los siguientes plugins en Jenkins:

1. **SonarQube Scanner for Jenkins**
   - Ve a: `Manage Jenkins` → `Manage Plugins` → `Available`
   - Busca "SonarQube Scanner"
   - Instala el plugin y reinicia Jenkins

## Paso 3: Configurar SonarQube Server en Jenkins

1. Ve a: `Manage Jenkins` → `Configure System`

2. Busca la sección **SonarQube servers**

3. Haz clic en **Add SonarQube**

4. Configura:
   - **Name**: `SonarQube` (debe coincidir con el nombre en el Jenkinsfile)
   - **Server URL**: `http://localhost:9000` (o la URL de tu servidor SonarQube)
   - **Server authentication token**: 
     - Crea un token en SonarQube: `My Account` → `Security` → `Generate Token`
     - En Jenkins, añade las credenciales: `Add` → `Jenkins` → `Secret text`
     - Pega el token de SonarQube
     - ID: `sonarqube-token`

5. Guarda los cambios

## Paso 4: Configurar SonarQube Scanner en Jenkins

1. Ve a: `Manage Jenkins` → `Global Tool Configuration`

2. Busca la sección **SonarQube Scanner**

3. Haz clic en **Add SonarQube Scanner**

4. Configura:
   - **Name**: `SonarQube Scanner` (debe coincidir con el nombre en el Jenkinsfile)
   - **Install automatically**: 
     - ✅ Marcar si quieres que Jenkins lo descargue automáticamente
     - O desmarca y especifica la ruta: `/opt/sonar-scanner`

5. Guarda los cambios

## Paso 5: Configurar Webhook en SonarQube (Opcional pero Recomendado)

Para que el Quality Gate funcione correctamente:

1. En SonarQube, ve a: `Administration` → `Configuration` → `Webhooks`

2. Crea un nuevo webhook:
   - **Name**: `Jenkins`
   - **URL**: `http://<JENKINS_URL>/sonarqube-webhook/`
   - Ejemplo: `http://localhost:8080/sonarqube-webhook/`

3. Guarda el webhook

## Paso 6: Verificar la Configuración

### Verificar Python en Jenkins

```bash
# SSH al servidor Jenkins
python3 --version
pip3 --version
```

### Verificar SonarQube Scanner

```bash
/opt/sonar-scanner/bin/sonar-scanner --version
```

## Parámetros de SonarQube en el Jenkinsfile

El Jenkinsfile configurado incluye los siguientes parámetros:

```groovy
-Dsonar.projectKey=cognitive-anomaly-detector
-Dsonar.projectName='Cognitive Anomaly Detector'
-Dsonar.projectVersion=1.0
-Dsonar.sources=src
-Dsonar.tests=tests
-Dsonar.python.coverage.reportPaths=coverage.xml
-Dsonar.python.xunit.reportPath=test-results/junit.xml
-Dsonar.exclusions=venv/**,htmlcov/**,*.pyc,__pycache__/**
```

Puedes modificar estos valores según tus necesidades en el archivo `Jenkinsfile`.

## Troubleshooting

### Error: "SonarQube server 'SonarQube' not found"

**Solución**: Verifica que el nombre del servidor en `Configure System` coincida exactamente con el nombre en `withSonarQubeEnv('SonarQube')`.

### Error: "Tool type 'hudson.plugins.sonar.SonarRunnerInstallation' does not have an install of 'SonarQube Scanner'"

**Solución**: Configura el SonarQube Scanner en `Global Tool Configuration` con el nombre exacto `SonarQube Scanner`.

### Error: "Quality Gate timeout"

**Solución**: 
1. Verifica que el webhook esté configurado correctamente en SonarQube
2. Verifica que SonarQube pueda alcanzar la URL de Jenkins
3. Aumenta el timeout en el Jenkinsfile si es necesario

### Error: "python3: command not found"

**Solución**: Instala Python 3 en el servidor Jenkins o actualiza la variable `PYTHON` en el Jenkinsfile para usar la ruta completa.

## Estructura del Pipeline

El pipeline modificado ejecuta las siguientes etapas:

1. **Checkout**: Descarga el código fuente
2. **Setup Environment**: Crea el entorno virtual Python e instala dependencias
3. **Code Quality Checks**: Lint y análisis de seguridad (en paralelo)
4. **Run Tests**: Ejecuta las pruebas con cobertura
5. **Test Coverage Report**: Genera y verifica el reporte de cobertura
6. **SonarQube Analysis**: Analiza el código con SonarQube
7. **Quality Gate**: Espera el resultado del Quality Gate de SonarQube
8. **Archive Artifacts**: Archiva los resultados y reportes

## Personalización

### Cambiar el umbral de cobertura en SonarQube

Puedes configurar umbrales de calidad en SonarQube:

1. Ve al proyecto en SonarQube
2. `Project Settings` → `Quality Gate`
3. Selecciona o crea un Quality Gate personalizado

### Ajustar exclusiones

Modifica el parámetro `-Dsonar.exclusions` en el Jenkinsfile para excluir archivos o directorios adicionales.

## Referencias

- [SonarQube Documentation](https://docs.sonarqube.org/)
- [SonarQube Scanner for Jenkins](https://docs.sonarqube.org/latest/analysis/scan/sonarscanner-for-jenkins/)
- [SonarQube Python Analysis](https://docs.sonarqube.org/latest/analysis/languages/python/)
