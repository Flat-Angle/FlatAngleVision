/*
	@version: 2.0
	@Author	: 肖思宇
*/

const rightBlock = document.querySelector(".right-block");
const fileInput = document.getElementById("identification_input");
const fileSubmit = document.getElementById("identification_submit");
const identificationInputButton = document.getElementById("browse_button");
const identificationSubmitButton = document.getElementById("upload_button");

// identificationInputButton.onclick = () =>
// {
// 	fileInput.click();
// }

// identificationSubmitButton.onclick = () =>
// {
// 	fileSubmit.click();
// }

function identificationInputFile()
{
	fileInput.click();
}


function identificationSubmitFile()
{
	fileSubmit.click();
}

// function showImg(e) {
// 	rightBlock.getElementById("overlay-warp").style.opacity=1;
// 	var img = document.createElement("img");
// 	img.setAttribute("class", "overlayimg");
// 	img.src = this.getAttribute("src");
// 	var clientWidth=document.documentElement.clientWidth;
// 	var clientHeight=document.documentElement.clientHeight;
// 	var imgWidth=this.width;
// 	var imgHeight=this.height;
// 	var percent=clientWidth/imgWidth;
// 	var realH=imgHeight*percent;
// 	var marginTop=0;
// 	if(realH>clientHeight){
// 		marginTop=clientHeight/2;
// 	}else {
// 		marginTop=realH/2;
// 	}
// 	img.style.marginTop=-marginTop+"px";
// 	var overlayEle = rightBlock.getElementById("overlay");
// 	overlayEle.appendChild(img);
// 	overlayEle.className ="overlay overlay-visible";
// }
// window.onload = function () {
// 	let imgs = rightBlock.getElementsByClassName("canShowBigImg");
// 	for (var i = 0; i < imgs.length; i++) {
// 		imgs[i].onclick = showImg;
// 	}
// 	var overlayEle = rightBlock.getElementById("overlay");
// 	overlayEle.onclick=function (e) {
// 		if(rightBlock.getElementsByClassName("overlayimg").length===1){
// 			var currentTarget=e.currentTarget;
// 			setTimeout(function (e) {
// 				currentTarget.removeChild(rightBlock.getElementsByClassName("overlayimg")[0]);
// 				rightBlock.getElementById("overlay-warp").style.opacity=0;
// 			},150);
// 			currentTarget.className="overlay";
// 		}
// 	}
//
// }


function showImg(e) {
	document.getElementById("overlay-warp").style.opacity=1;
	var img = document.createElement("img");
	img.setAttribute("class", "overlayimg");
	img.src = this.getAttribute("src");
	var clientWidth=document.documentElement.clientWidth;
	var clientHeight=document.documentElement.clientHeight;
	var imgWidth=this.width;
	var imgHeight=this.height;
	var percent=clientWidth/imgWidth;
	var realH=imgHeight*percent;
	var marginTop=0;
	if(realH>clientHeight){
	marginTop=clientHeight/2;
}else {
	marginTop=realH/2;
}
	img.style.marginTop=-marginTop+"px";
	var overlayEle = document.getElementById("overlay");
	overlayEle.appendChild(img);
	overlayEle.className ="overlay overlay-visible";
}
	window.onload = function () {
	var imgs = document.getElementsByClassName("canShowBigImg");
	for (var i = 0; i < imgs.length; i++) {
	imgs[i].onclick = showImg;
}
	var overlayEle = document.getElementById("overlay");
	overlayEle.onclick=function (e) {
	if(document.getElementsByClassName("overlayimg").length==1){
	var currentTarget=e.currentTarget;
	setTimeout(function (e) {
	currentTarget.removeChild(document.getElementsByClassName("overlayimg")[0]);
	document.getElementById("overlay-warp").style.opacity=0;
},150);
	currentTarget.className="overlay";
}
}

}
