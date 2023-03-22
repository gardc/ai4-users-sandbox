
# Stop the Docker Compose if it's already running
ssh root@64.226.100.80 << EOF
    "cd ai4users-sandbox/ && docker-compose down"
EOF

cd ..

# Upload the local files to the server
rsync -ravz --delete ai4-users-sandbox root@64.226.100.80:ai4users-sandbox

# Start the Docker Compose on the server
ssh root@64.226.100.80 << EOF
    "cd ai4users-sandbox/ && docker-compose build && docker-compose up -d"
EOF