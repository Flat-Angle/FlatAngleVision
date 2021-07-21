/*
	@version: 2.0
	@Author	: 肖思宇
*/

const dropArea = document.querySelector(".drag_area");
const dragText = dropArea.querySelector("header");
const button = dropArea.querySelector("button");
const input1 = document.getElementById("inputfile");
const input2 = document.getElementById("uploadfile");
const img = dropArea.querySelector("img");
const buttonArea = document.querySelector('.button_area');
// const buttonArea = document.getElementById("button_area");
const uploadButton = buttonArea.querySelector("button");
// const input2 = buttonArea.querySelector('input');

const inputFile = document.getElementById('inputfile');

let file;//文件
let filePath;
let fileURL;
let fileReader;

//获取地址
inputFile.onchange = function ()
{
    let url = null;
    let fileObj = document.getElementById("inputfile").files[0];
    if (window.createObjcectURL !== undefined) {
        url = window.createOjcectURL(fileObj);
    } else if (window.URL !== undefined) {
        url = window.URL.createObjectURL(fileObj);
    } else if (window.webkitURL !== undefined) {
        url = window.webkitURL.createObjectURL(fileObj);
    }
    console.log(url)
}

//点击按钮调用隐藏的input的点击效果
button.onclick = () =>
{
    input1.click();
    // uploadFile();
}

function uploadFile() {

    // 找到文件文件选择框
    var fileInput = document.querySelector("#fileSelecter");

    // 获取选择的文件
    // (因为input是支持选择多个文件的，所以获取文件通过files字段，如果单个文件也是在这个files列表里。)
    var file = fileInput.files.item(0);

    // 判断一下
    if (file == null) {
        // 没有选择文件。就什么都不处理。
        return;
    }

    // 使用FileReader读取文件。
    let fileReader = new FileReader();

    fileReader.addEventListener("error", function (ev) {
        // 文件读取出错时，执行此方法。
        // 通过 fileReader.error 可以获取到错误信息。
    });

    fileReader.addEventListener("load", function (ev) {
        // 文件读取成功后调用此方法。
        // 通过 fileReader.result 即可获取到文件内容。
        var result = fileReader.result;
        console.log("hello")
        console.log(result)
        $.post("http://121.199.2.56:5000/segmentation/upload_file", {
            base64Data: result
        }, function (response) {
            // 服务器响应了我们的上传请求。
        });
    });

    fileReader.addEventListener("loadstart", function (ev) {
        // 读取开始时此方法被调用。
    });

    fileReader.addEventListener("loadend", function (ev) {
        // 文件读取结束时执行此方法。
        // 无论读取成功，还是读取失败。
        // 总之，在结束读文件操作时，此方法都会调用。
    });

    fileReader.addEventListener("abort", function (ev) {
        // 文件读取被中断时，此方法调用。
        // 你可以通过 fileReader.abort() 方法随时中断文件的读取。
    });

    fileReader.addEventListener("progress", function (ev) {
        // 读取文件过程不是一次性读完的，会进行多次读取。
        // 没读取一次，本方法执行一次。
    });

    // 将文件内容读取为 base64 内容。通过 fileReader.result 即可返回base64的数据内容。
    fileReader.readAsDataURL(file);
}

uploadButton.onclick = () =>
{
    input2.click();

}

fileReader.addEventListener("load", function (ev) {
    // 文件读取成功后调用此方法。
    // 通过 fileReader.result 即可获取到文件内容。
    var result = fileReader.result;
    $.post("http://121.199.2.56:5000/segmentation/upload_file", {
        base64Data: result
    }, function (response) {
        console.log("NMSL")
        // 服务器响应了我们的上传请求。
    });
});


function UploadFile()
{
    console.log(file);
    uploadButton.style.display = "none";
    buttonArea.classList.add("active");
}

// uploadButton.onclick = () =>
// {
// 	// input2.click();
// 	UploadFile();
// }


//若用户将图片选定
//加事件监听器
input1.addEventListener("change", function(){
    file = this.files[0];
    dropArea.classList.add("active");
    showFile();
    popButton();
    dropArea.classList.add("showFile");
})

//若用户将图片拖拽至选定区域内
//加事件监听器
dropArea.addEventListener("dragover",(event)=>{
    event.preventDefault();
    // console.log("File is over DragArea");
    dropArea.classList.add("active");
    //dropArea.style.backgroundColor = #ddd;
    // function change()
    // {
    // 	dropArea.style.backgroundColor = #ddd;
    // }
    dragText.textContent = "Release to Upload File";
})

//若用户将图片拖离选定区域内
//加事件监听器
dropArea.addEventListener("dragleave",()=>{
    // console.log("File is outside from DragArea");
    dropArea.classList.remove("active");
    dragText.textContent = "Drag & Drop To Upload File";
})

//若用户将图片拖放至选定区域内
//加事件监听器
dropArea.addEventListener("drop",(event)=>{
    event.preventDefault();
    // console.log("File is dropped on DragArea");
    // 获取用户选择文件和[0]这意味着如果用户选择多个文件，
    // 那么我们只选择第一个
    file = event.dataTransfer.files[0];
    showFile();
    popButton();
    dropArea.classList.add("showFile");
})

//图片预览
function showFile()
{
    let fileType = file.type;
    console.log(fileType);




    // var src = event.target || window.event.srcElement; //获取事件源，兼容chrome/IE
    // filePath = src.value;//图片完整路径
    // filePath = getFilePath(input1);
    // console.log("this is me");

    //吐了
    // var url;
    // var name = document.getElementById("file").files[0];
    // if (window.URL != undefined && name) {
    // 	url = window.URL.createObjectURL(name);
    // }
    // console.log(document.getElementById("file").value);
    // console.log(url);

    // console.log(filePath);

    // let vaildExtensions = ["image/png","image/jpg","image/jpeg"];



    if (!fileType.startsWith("image/")) {
        alert("输入的不是图片！");
        dropArea.classList.remove("active");
    } else{
        console.log("This is an image!");
        //创建图片预览所需的FileReader对象
        fileReader = new FileReader();
        // console.log(fileReader.readAsDataURL(file));
        fileReader.onload = ()=>{
            //传入URL
            fileURL = fileReader.result;

            console.log(fileURL);
            let imgTag = `<img src = "${fileURL}"  alt = "">`;
            //图片预览
            dropArea.innerHTML = imgTag;
        }
        fileReader.readAsDataURL(file);
        dropArea.classList.add("showFile");
    }
}

//获取真实地址
function getFilePath(input){
    if(input){//input是<input type="file">Dom对象
        if(window.navigator.userAgent.indexOf("MSIE")>=1){  //如果是IE
            input.select();
            return document.selection.createRange().text;
        }
        else if(window.navigator.userAgent.indexOf("Firefox")>=1){  //如果是火狐  {
            if(input.files){
                return input.files.item(0).getAsDataURL();
            }
            return input.value;
        }
        return input.value;
    }
}


//弹入按钮
function popButton()
{
    buttonArea.style.display = "block";

    // let button1 = `<button type="button"> Browse File </button>`;
    // buttonArea.innerHTML = button1;
    // buttonArea.innerHTML = "<br/>";

    // buttonArea.append(button1);
    // buttonArea.append(button2);

    // var button1 = buttonArea.createElement("button");
    // input.setAttribute('type', 'button');
    // button1.innerText("Browse File");
    // buttonArea.appendChild(button1);

    // var br = buttonArea.createElement("br");
    // buttonArea.appendChild(br);

    // var button2 = buttonArea.createElement("button");
    // input.setAttribute('type', 'button');
    // button2.innerText("Browse File");
    // buttonArea.appendChild(button2);

}