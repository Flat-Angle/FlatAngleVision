/*
	@version: 2.0
	@Author	: 肖思宇
*/

$(window).on('scroll',function(){
			if($(window).scrollTop() > 900){//这里100代表你要动画的元素离最顶层的距离，console.log一下就知道了。
				$('.block_type_1_img').addClass('active')
				$('.block_type_1_words').addClass('active')
			}
			if($(window).scrollTop() > 1800){//这里100代表你要动画的元素离最顶层的距离，console.log一下就知道了。
				$('.block_type_2_img').addClass('active')
				$('.block_type_2_words').addClass('active')
			}
			if($(window).scrollTop() > 2400){
				$('.block_type_3_img').addClass('active')
				$('.block_type_3_words').addClass('active')
			}
		});

// function _scroll(){
//             var scrollTop =$(window).scrollTop();
//             if(scrollTop > 900){
//                 $('.block_type_1_img').css('-webkit-animation',right-to-left 1s linear);
//             }else{
//                 $('.header').css('opacity',1);
//             }
//         }
//         $(window).on('scroll',function() {
//             _scroll()
//         });