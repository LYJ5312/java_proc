package com.spring.test;

import org.springframework.context.ApplicationContext;
import org.springframework.context.support.GenericXmlApplicationContext;

public class Main {

	public static void main(String[] args)throws Exception {
		
		//C c = new C();
		
		ApplicationContext context 
			= new GenericXmlApplicationContext("classpath:com/spring/context/application.xml");
		//classpath를 쓰면 웤스페이스의 프로젝트 경로까지 알아서 가져옴
		
		C c = context.getBean("c",C.class);
		
		
		Z z = c.getZ();
		
		z.a();

	}

}
