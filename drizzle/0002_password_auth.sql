CREATE TABLE `passwordCredentials` (
	`userId` int NOT NULL,
	`passwordHash` varchar(255) NOT NULL,
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `passwordCredentials_userId` PRIMARY KEY(`userId`)
);
--> statement-breakpoint
ALTER TABLE `passwordCredentials` ADD CONSTRAINT `passwordCredentials_userId_users_id_fk` FOREIGN KEY (`userId`) REFERENCES `users`(`id`) ON DELETE cascade ON UPDATE no action;
--> statement-breakpoint
CREATE INDEX `users_email_idx` ON `users` (`email`);
