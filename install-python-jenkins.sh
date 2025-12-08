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
if ! apt-get update; then
    echo "Error: Failed to update repositories" >&2
    exit 1
fi

# Instalar Python 3 y herramientas necesarias
echo ""
echo "Instalando Python 3 y dependencias..."
if ! apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential; then
    echo "Error: Failed to install Python packages" >&2
    exit 1
fi

# Verificar instalación
echo ""
echo "Verificando instalación..."
if ! python3 --version; then
    echo "Error: Python3 installation verification failed" >&2
    exit 1
fi

if ! pip3 --version; then
    echo "Error: pip3 installation verification failed" >&2
    exit 1
fi

echo ""
echo "============================================"
echo "✓ Python 3 instalado correctamente"
echo "============================================"
echo ""
echo "Comandos disponibles:"
echo "  - python3"
echo "  - pip3"
echo ""
