#!/bin/bash

# Script para instalar Python en el contenedor de Jenkins
# Este script debe ejecutarse con permisos de root

set -e

echo "============================================"
echo "Instalación de Python 3 en Jenkins"
echo "============================================"

# Detectar si estamos en un contenedor
if [ -f /.dockerenv ]; then
    echo "✓ Ejecutando dentro de un contenedor Docker"
else
    echo "⚠ No se detectó contenedor Docker. Continuando de todas formas..."
fi

# Actualizar repositorios
echo ""
echo "Actualizando repositorios..."
apt-get update

# Instalar Python 3 y herramientas necesarias
echo ""
echo "Instalando Python 3 y dependencias..."
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential

# Verificar instalación
echo ""
echo "Verificando instalación..."
python3 --version
pip3 --version

echo ""
echo "============================================"
echo "✓ Python 3 instalado correctamente"
echo "============================================"
echo ""
echo "Comandos disponibles:"
echo "  - python3"
echo "  - pip3"
echo ""
