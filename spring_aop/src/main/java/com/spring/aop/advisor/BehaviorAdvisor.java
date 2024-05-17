package com.spring.aop.advisor;

import org.aspectj.lang.ProceedingJoinPoint;

public class BehaviorAdvisor {
	public void chikachika() {
		System.out.println("양치질");
	}
	
	public void chikachikaAround(ProceedingJoinPoint joinPoint) throws Throwable{
			System.out.println("한번 닦음");
			joinPoint.proceed();
			System.out.println("또 닦아");
	}
}
