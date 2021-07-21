/*
	@version: 2.0
	@Author	: 肖思宇
*/

const previewButton = document.querySelector(".preview_button");
const uploadButton = document.querySelector(".upload_button");
const button1 = document.getElementById("input");
const button2 = document.getElementById("upload");
// const button3 = document.getElementById("select");

const downloadButton = document.getElementById("download_button");
const downImg = document.getElementById("download_img");


let segmentationFile;

//点击按钮previewButton调用隐藏的button1的点击效果
previewButton.onclick = () =>
{
    button1.click();
}

//点击按钮uploadButton调用隐藏的button2的点击效果
function inputFile()
{
    button2.click();

}

downloadButton.onclick = () =>
{

	alert("我是伞兵");
	alert(downImg.src);
	downloadFile();

}


// 非跨域
function downloadFile()
{
    alert(downImg.src);
    let url = downImg.src;
    let a = document.createElement('a');              // 创建一个a节点插入的document
    let event = new MouseEvent('click');               // 模拟鼠标click点击事件
	a.download = 'download';                           // 设置a节点的download属性值,图片名称
	a.href = url;                                     // 将图片的src赋值给a节点的href
	a.dispatchEvent(event);
    window.URL.revokeObjectURL(url);
}

//若用户将图片选定
//加事件监听器
button1.addEventListener("change", function(){
    segmentationFile = this.files[0];
    segmentationShowFile();
})

//图片预览
function segmentationShowFile()
{
    let fileType = segmentationFile.type;

    console.log(fileType);

    if (!fileType.startsWith("image/")) {
        alert("输入的不是图片！");
    } else{
        console.log("This is an image!");
        //创建图片预览所需的FileReader对象
        fileReader = new FileReader();
        fileReader.onload = ()=>{
            //传入URL
            fileURL = fileReader.result;

            console.log(fileURL);
            document.getElementById("segmentationShowImage").src = `${fileURL}`;

        }
        fileReader.readAsDataURL(file);
    }
}


// // 跨域下载方案二
// function downloadPic() {
//     let imgsrc= downImg.src;
//     let x = new XMLHttpRequest()
//     x.open('GET', imgsrc, true)
//     x.responseType = 'blob'
//     x.onload = function () {
//         let url = window.URL.createObjectURL(x.response)
//         let a = document.createElement('a')
//         a.href = url
//         a.download = 'download';
//         a.click()
//     }
//     x.send()
// }

// 跨域下载方案一 base64编码形式
// function downloadIamge(imgsrc, name) {//下载图片地址和图片名
//     let image = new Image();
//     // 解决跨域 Canvas 污染问题
//     image.setAttribute("crossOrigin", "anonymous");
//     image.onload = function() {
//         let canvas = document.createElement("canvas");
//         canvas.width = image.width;
//         canvas.height = image.height;
//         let context = canvas.getContext("2d");
//         context.drawImage(image, 0, 0, image.width, image.height);
//         let url = canvas.toDataURL("image/png"); //得到图片的base64编码数据
//         let a = document.createElement("a"); // 生成一个a元素
//         let event = new MouseEvent("click"); // 创建一个单击事件
//         a.download = name || "photo"; // 设置图片名称
//         a.href = url; // 将生成的URL设置为a.href属性
//         a.dispatchEvent(event); // 触发a的单击事件
//     };
//     image.src = imgsrc;
// }




// // downloadByBlob('链接地址', '下载后的文件命名')
// function downloadByBlob (name) {
// 	let url = downImg.src;
// 	console.log("你是伞兵")
// 	let image = new Image() // 创建一个image标签
// 	image.setAttribute('crossOrigin', 'anonymous') // 设置属性
// 	image.src = url // 设置src
// 	// 加载图片，缓存到本地的canvas中（避免图片地址导致的下载跨域）
// 	image.onload = () => {
// 		let canvas = document.createElement('canvas')
// 		canvas.width = image.width
// 		canvas.height = image.height
// 		let ctx = canvas.getContext('2d')
// 		ctx.drawImage(image, 0, 0, image.width, image.height) // 画布中绘制图片
// 		canvas.toBlob((blob) => {
// 			let url = URL.createObjectURL(blob)
// 			this.download(url, name)
// 			URL.revokeObjectURL(url) // 用完释放URL对象
// 		})
// 	}
// }
//
// function download (href, name) {
//   let eleLink = document.createElement('a') // 创建一个a标签
//   eleLink.download = name // 下载命名
//   eleLink.href = href // 下载地址
//   eleLink.click() // 模拟点击
//   eleLink.remove() // 模拟点击移除
// }