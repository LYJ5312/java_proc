package com.spring.task;

import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

//@Component("jobTask")
public class TestScheduler {

//	@Scheduled(cron = "*/5 * * * * *")
	public void testMessage() {
		System.out.println("안녕");
	} 
}
