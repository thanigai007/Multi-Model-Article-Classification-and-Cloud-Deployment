CREATE DATABASE user_db;

USE user_db;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
	login_time DATETIME DEFAULT CURRENT_TIMESTAMP
);


drop database user_db;

select * from users;