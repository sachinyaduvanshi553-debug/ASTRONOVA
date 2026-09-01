$ErrorActionPreference = 'Stop'
$InfraDir = "$PWD\infra"
if (!(Test-Path $InfraDir)) { New-Item -ItemType Directory -Force -Path $InfraDir }

Write-Host "1. Setting up Portable Redis..."
$RedisDir = "$InfraDir\redis"
if (!(Test-Path "$RedisDir\redis-server.exe")) {
    Write-Host "Downloading Redis..."
    Invoke-WebRequest -Uri "https://github.com/tporadowski/redis/releases/download/v5.0.14.1/Redis-x64-5.0.14.1.zip" -OutFile "$InfraDir\redis.zip"
    Expand-Archive -Path "$InfraDir\redis.zip" -DestinationPath $RedisDir -Force
    Remove-Item "$InfraDir\redis.zip"
}

Write-Host "2. Setting up Portable PostgreSQL..."
$PgDir = "$InfraDir\pgsql"
if (!(Test-Path "$PgDir\bin\pg_ctl.exe")) {
    Write-Host "Downloading PostgreSQL..."
    Invoke-WebRequest -Uri "https://get.enterprisedb.com/postgresql/postgresql-10.23-1-windows-x64-binaries.zip" -OutFile "$InfraDir\pgsql.zip"
    Expand-Archive -Path "$InfraDir\pgsql.zip" -DestinationPath $InfraDir -Force
    Remove-Item "$InfraDir\pgsql.zip"
    
    Write-Host "Initializing PostgreSQL Database..."
    $PassFile = "$InfraDir\pg_pass.txt"
    Set-Content -Path $PassFile -Value "changeme_secure_password"
    & "$PgDir\bin\initdb.exe" -U astronova -D "$PgDir\data" --pwfile="$PassFile"
}

Write-Host "3. Setting up Portable Java (for Kafka)..."
$JdkDir = "$InfraDir\jdk"
if (!(Test-Path "$JdkDir\bin\java.exe")) {
    Write-Host "Downloading OpenJDK..."
    Invoke-WebRequest -Uri "https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_windows-x64_bin.zip" -OutFile "$InfraDir\jdk.zip"
    Expand-Archive -Path "$InfraDir\jdk.zip" -DestinationPath $InfraDir -Force
    Rename-Item "$InfraDir\jdk-17.0.2" "jdk"
    Remove-Item "$InfraDir\jdk.zip"
}

Write-Host "4. Setting up Portable Kafka..."
$KafkaDir = "$InfraDir\kafka"
if (!(Test-Path "$KafkaDir\bin\windows\kafka-server-start.bat")) {
    Write-Host "Downloading Kafka..."
    Invoke-WebRequest -Uri "https://archive.apache.org/dist/kafka/3.6.1/kafka_2.13-3.6.1.tgz" -OutFile "$InfraDir\kafka.tgz"
    Write-Host "Extracting Kafka..."
    cmd /c "tar -xf $InfraDir\kafka.tgz -C $InfraDir"
    Rename-Item "$InfraDir\kafka_2.13-3.6.1" "kafka"
    Remove-Item "$InfraDir\kafka.tgz"
}

Write-Host "Starting Services..."
Write-Host "Starting Redis..."
Start-Process -FilePath "$RedisDir\redis-server.exe" -WindowStyle Minimized

Write-Host "Starting Postgres..."
Start-Process -FilePath "$PgDir\bin\pg_ctl.exe" -ArgumentList "-D", "$PgDir\data", "start" -WindowStyle Minimized
Start-Sleep -Seconds 4
Write-Host "Creating astronova database..."
& "$PgDir\bin\psql.exe" -U astronova -d postgres -c "CREATE DATABASE astronova;"
& "$PgDir\bin\psql.exe" -U astronova -d astronova -f scripts\init_db_native.sql

Write-Host "Starting Zookeeper..."
$env:JAVA_HOME = "$JdkDir"
$env:PATH = "$JdkDir\bin;" + $env:PATH
Start-Process -FilePath "$KafkaDir\bin\windows\zookeeper-server-start.bat" -ArgumentList "$KafkaDir\config\zookeeper.properties" -WindowStyle Minimized
Start-Sleep -Seconds 5

Write-Host "Starting Kafka..."
Start-Process -FilePath "$KafkaDir\bin\windows\kafka-server-start.bat" -ArgumentList "$KafkaDir\config\server.properties" -WindowStyle Minimized
Start-Sleep -Seconds 3

Write-Host "All native infrastructure started!"
