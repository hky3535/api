// 选择框架选项
const templateSelect = document.getElementById("template_select");
// 框架参数模板
const template = document.querySelectorAll(".template");
// 文件上传文件
const weightInput = document.getElementById("weight_input");
const configInput = document.getElementById("config_input");
const frameInput = document.getElementById("frame_input");
const weightUpload = document.getElementsByClassName("weight_upload");
const configUpload = document.getElementsByClassName("config_upload");
const frameUpload = document.getElementsByClassName("frame_upload");
const weightNameStatus = document.querySelectorAll(".weight_name_status");
const configNameStatus = document.querySelectorAll(".config_name_status");
const frameNameStatus = document.querySelectorAll(".frame_name_status");
// 加载推理按钮
const loadButton = document.getElementById("load_button");
const inferButton = document.getElementById("infer_button");
// 模型推理 模型选择
const weightNameSelect = document.getElementById("weight_name_select");
// 消息提示
const printSpan = document.getElementById("print_span");
// 推理结果
const frameCanvasDiv = document.getElementById("frame_canvas_div");
const frameCanvas = document.getElementById("frame_canvas");
const frameContext = frameCanvas.getContext('2d');
// 拖拽上传
const frameDiv = document.getElementById("frame_div");
const frameDragoverUploadDiv = document.getElementById("frame_dragover_upload_div");

function print(content) {
    printSpan.textContent = content;
}

function post(url, data) {
    return new Promise((resolve, reject) => {
        fetch(url, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(response => {resolve(response);})
        .catch(error => {resolve(error);});
    });
}

function cvtFileBase64(fileInput) {
    return new Promise((resolve, reject) => {
        if (fileInput.files.length === 0) {resolve(false);} // 如果没有文件则直接返回false
        var file = fileInput.files[0]; // 如果选择了多个文件则默认为第一个
        var reader = new FileReader();
        reader.onload = function(e) {
            var fileBase64 = e.target.result;
            fileBase64 = fileBase64.split(',')[1]; // 去除base64头
            resolve({"name": file.name, "b64": fileBase64});
        };
        reader.readAsDataURL(file);
    });
}

async function weightNameSelectRefresh(template) {
    weightNameSelect.innerHTML = '';
    var status = await post("/status", {}); console.info(status);
    status = status.response[template];
    status.forEach(status => {
        var option = document.createElement('option');
        option.textContent = status;
        weightNameSelect.appendChild(option);
    });
}

function drawResults(frameBase64, results) {
    const image = new Image();
    image.onload = function() {
        // 归一化图像到canvas的尺寸
        var inputWidth = image.width; var inputHeight = image.height;
        var outputWidth = frameCanvasDiv.clientWidth; var outputHeight = frameCanvasDiv.clientHeight;

        var resizeWidthRate = 1; var resizeHeightRate = 1;
        if (inputWidth > outputWidth) {resizeWidthRate = outputWidth / inputWidth;}
        if (inputHeight > outputHeight) {resizeHeightRate = outputHeight / inputHeight;}
        var resizeRate = Math.min(resizeWidthRate, resizeHeightRate);

        frameCanvas.width = inputWidth * resizeRate;
        frameCanvas.height = inputHeight * resizeRate;
        
        frameContext.clearRect(0, 0, outputWidth, outputHeight);
        frameContext.drawImage(image, 0, 0, frameCanvas.width, frameCanvas.height);

        frameContext.strokeStyle = 'blue';
        frameContext.fillStyle = 'blue';
        results.forEach(result => {
            var [x0, y0, x1, y1, conf, cls] = result;
            var x = x0 * resizeRate; var y = y0 * resizeRate; var w = (x1 - x0) * resizeRate; var h = (y1 - y0) * resizeRate;
            frameContext.strokeRect(x, y, w, h);
            frameContext.fillText(`${cls} ${conf}%`, x, y-2);
        });
    };
    image.src = frameBase64;
}

async function load(templateName) {
    print(`框架：${templateName}；模型加载中`);

    var weight = await cvtFileBase64(weightInput);
    var config = await cvtFileBase64(configInput);

    // 检查模型是否已经加载
    var status = await post("/status", {}); console.info(status);
    if (status.response[templateName].includes(weight.name)) {
        print(`框架：${templateName}；模型：${weight.name}已存在，无需重复加载！`);
        return false;
    }

    var data = {}; // 加载参数json字段
    
    if (templateName === "grounding_dino") {
        if (weight === false) {print("请上传模型文件"); return false;}
        if (config === false) {print("请上传参数文件"); return false;}
        print(`已选择模型文件：${weight.name}；参数文件：${config.name}`);
        var device = document.querySelector(`.${templateName} .device_select`).value;
        data = {
            "engine_name": templateName, 
            "engine_task": weight.name, 
            "arguments": {
                "weight": {"name": weight.name, "b64": weight.b64},
                "config": {"name": config.name, "b64": config.b64}, 
                "device": device
            }
        }
    } else if (templateName === "yolo_worldu") {
        if (weight === false) {print("请上传模型文件"); return false;}
        print(`已选择模型文件：${weight.name}`);
        var device = document.querySelector(`.${templateName} .device_select`).value;
        var half = (document.querySelector(`.${templateName} .half_select`).value === 'true');
        data = {
            "engine_name": templateName, 
            "engine_task": weight.name, 
            "arguments": {
                "weight": {"name": weight.name, "b64": weight.b64},
                "device": device, 
                "half": half
            }
        }
    } else if (templateName == "yolov4") {
        if (weight === false) {print("请上传模型文件"); return false;}
        if (config === false) {print("请上传参数文件"); return false;}
        print(`已选择模型文件：${weight.name}；参数文件：${config.name}`);
        var device = document.querySelector(`.${templateName} .device_select`).value;
        data = {
            "engine_name": templateName, 
            "engine_task": weight.name, 
            "arguments": {
                "weight": {"name": weight.name, "b64": weight.b64},
                "config": {"name": config.name, "b64": config.b64}, 
                "device": device
            }
        }
    } else if (templateName == "yolov5") {
        if (weight === false) {print("请上传模型文件"); return false;}
        print(`已选择模型文件：${weight.name}`);
        var device = document.querySelector(`.${templateName} .device_select`).value;
        var half = (document.querySelector(`.${templateName} .half_select`).value === 'true');
        data = {
            "engine_name": templateName, 
            "engine_task": weight.name, 
            "arguments": {
                "weight": {"name": weight.name, "b64": weight.b64},
                "device": device, 
                "half": half
            }
        }
    } else if (templateName == "yolov5u") {
        if (weight === false) {print("请上传模型文件"); return false;}
        print(`已选择模型文件：${weight.name}`);
        var device = document.querySelector(`.${templateName} .device_select`).value;
        var half = (document.querySelector(`.${templateName} .half_select`).value === 'true');
        data = {
            "engine_name": templateName, 
            "engine_task": weight.name, 
            "arguments": {
                "weight": {"name": weight.name, "b64": weight.b64},
                "device": device, 
                "half": half
            }
        }
    } else if (templateName == "yolov8u") {
        if (weight === false) {print("请上传模型文件"); return false;}
        print(`已选择模型文件：${weight.name}`);
        var device = document.querySelector(`.${templateName} .device_select`).value;
        var half = (document.querySelector(`.${templateName} .half_select`).value === 'true');
        data = {
            "engine_name": templateName, 
            "engine_task": weight.name, 
            "arguments": {
                "weight": {"name": weight.name, "b64": weight.b64},
                "device": device, 
                "half": half
            }
        }
    }

    print(`框架：${templateName}；模型：${weight.name}已加载完成！上传中`);
    var response = await post("/load", data); console.info(response); // 请求后端以进行加载
    weightNameSelectRefresh(templateName); // 显示对应的已加载模型
    print(`框架：${templateName}；模型：${weight.name}已上传完成！`);
}
async function infer(templateName, weightName) {
    var frame = await cvtFileBase64(frameInput);

    var data = {}; // 推理参数json字段

    if (templateName === "grounding_dino") {
        if (frame === false) {print("请上传图像文件"); return false;}
        print(`已选择图像文件：${frame.name}`);
        var conf = parseFloat(document.querySelector(`.${templateName} .conf_input`).value);
        var iou = parseFloat(document.querySelector(`.${templateName} .iou_input`).value);
        var classes = document.querySelector(`.${templateName} .classes_input`).value;
        if (classes === "false") {classes = false;} else {classes = JSON.parse(classes);}
        data = {
            "engine_name": templateName, 
            "engine_task": weightName,
            "arguments": {
                "frame": {"b64": frame.b64}, 
                "conf": conf, 
                "iou": iou, 
                "classes": classes, 
            }
        }
    } else if (templateName === "yolo_worldu") {
        if (frame === false) {print("请上传图像文件"); return false;}
        print(`已选择图像文件：${frame.name}`);
        var conf = parseFloat(document.querySelector(`.${templateName} .conf_input`).value);
        var iou = parseFloat(document.querySelector(`.${templateName} .iou_input`).value);
        var classes = document.querySelector(`.${templateName} .classes_input`).value;
        if (classes === "false") {classes = false;} else {classes = JSON.parse(classes);}
        data = {
            "engine_name": templateName, 
            "engine_task": weightName,
            "arguments": {
                "frame": {"b64": frame.b64}, 
                "conf": conf, 
                "iou": iou, 
                "classes": classes
            }
        }
    } else if (templateName == "yolov4") {
        if (frame === false) {print("请上传图像文件"); return false;}
        print(`已选择图像文件：${frame.name}`);
        var conf = parseFloat(document.querySelector(`.${templateName} .conf_input`).value);
        var iou = parseFloat(document.querySelector(`.${templateName} .iou_input`).value);
        var classes = document.querySelector(`.${templateName} .classes_input`).value;
        if (classes === "false") {classes = false;} else {classes = JSON.parse(classes);}
        data = {
            "engine_name": templateName, 
            "engine_task": weightName, 
            "arguments": {
                "frame": {"b64": frame.b64}, 
                "conf": conf, 
                "iou": iou, 
                "classes": classes
            }
        }
    } else if (templateName == "yolov5") {
        if (frame === false) {print("请上传图像文件"); return false;}
        print(`已选择图像文件：${frame.name}`);
        var conf = parseFloat(document.querySelector(`.${templateName} .conf_input`).value);
        var iou = parseFloat(document.querySelector(`.${templateName} .iou_input`).value);
        var classes = document.querySelector(`.${templateName} .classes_input`).value;
        if (classes === "false") {classes = false;} else {classes = JSON.parse(classes);}
        data = {
            "engine_name": templateName, 
            "engine_task": weightName, 
            "arguments": {
                "frame": {"b64": frame.b64}, 
                "conf": conf, 
                "iou": iou, 
                "classes": classes
            }
        }
    } else if (templateName == "yolov5u") {
        if (frame === false) {print("请上传图像文件"); return false;}
        print(`已选择图像文件：${frame.name}`);
        var conf = parseFloat(document.querySelector(`.${templateName} .conf_input`).value);
        var iou = parseFloat(document.querySelector(`.${templateName} .iou_input`).value);
        var classes = document.querySelector(`.${templateName} .classes_input`).value;
        if (classes === "false") {classes = false;} else {classes = JSON.parse(classes);}
        data = {
            "engine_name": templateName, 
            "engine_task": weightName, 
            "arguments": {
                "frame": {"b64": frame.b64}, 
                "conf": conf, 
                "iou": iou, 
                "classes": classes
            }
        }
    } else if (templateName == "yolov8u") {
        if (frame === false) {print("请上传图像文件"); return false;}
        print(`已选择图像文件：${frame.name}`);
        var conf = parseFloat(document.querySelector(`.${templateName} .conf_input`).value);
        var iou = parseFloat(document.querySelector(`.${templateName} .iou_input`).value);
        var classes = document.querySelector(`.${templateName} .classes_input`).value;
        if (classes === "false") {classes = false;} else {classes = JSON.parse(classes);}
        data = {
            "engine_name": templateName, 
            "engine_task": weightName, 
            "arguments": {
                "frame": {"b64": frame.b64}, 
                "conf": conf, 
                "iou": iou, 
                "classes": classes
            }
        }
    }

    print(`框架：${templateName}；模型：${weightName}；图片：${frame.name}推理中`);
    var response = await post("/infer", data); console.info(response); // 请求后端以进行推理
    drawResults("data:image/png;base64," + frame.b64, response.response); // 显示推理结果
    print(`框架：${templateName}；模型：${weightName}；图片：${frame.name}已推理完成！`);
}

function initEvent() {
    // 选择框架选项 选择后切换框架参数模板
    function showTemplate(templateName) {
        template.forEach(function(element) { // 遍历所有框架参数模板
            if (element.classList.contains(templateName)) {element.style.display = "flex";} 
            else {element.style.display = "none";} // 选择其中一个进行显示
        });
    }
    templateSelect.addEventListener("change", async function() {
        print(`选择框架：${templateSelect.value}`);
        // 每次切换框架参数模板都清空所有的文件选择
        weightInput.value = null;
        configInput.value = null;
        // frameInput.value = null;
        weightNameStatus.forEach(function(element) {element.textContent = "未上传";});
        configNameStatus.forEach(function(element) {element.textContent = "未上传";});
        // frameNameStatus.forEach(function(element) {element.textContent = "未上传";});
        // 显示对应的参数模板组件
        showTemplate(templateSelect.value);
        // 显示对应的已加载模型
        weightNameSelectRefresh(templateSelect.value)
    });
    print(`选择框架：${templateSelect.value}`);
    // 绑定所有模型上传按钮
    for (var i = 0; i < weightUpload.length; i++) {
        var weightUploadButton = weightUpload[i];
        weightUploadButton.addEventListener("click", function() {
            weightInput.click();
        });
    }
    weightInput.addEventListener('change', function(event) {
        weightNameStatus.forEach(function(element) { // 遍历所有框架参数模板
            element.textContent = event.target.files[0].name;
        });
    });
    // 绑定所有参数上传按钮
    for (var i = 0; i < configUpload.length; i++) {
        var configUploadButton = configUpload[i];
        configUploadButton.addEventListener("click", function() {
            configInput.click();
        });
    }
    configInput.addEventListener('change', function(event) {
        configNameStatus.forEach(function(element) { // 遍历所有框架参数模板
            element.textContent = event.target.files[0].name;
        });
    });
    // 绑定所有图像上传按钮
    for (var i = 0; i < frameUpload.length; i++) {
        var frameUploadButton = frameUpload[i];
        frameUploadButton.addEventListener("click", function() {
            frameInput.click();
        });
    }
    frameInput.addEventListener('change', function(event) {
        frameNameStatus.forEach(function(element) { // 遍历所有框架参数模板
            element.textContent = event.target.files[0].name;
        });
    });
    // 绑定加载推理按钮
    loadButton.addEventListener("click", function() {load(templateSelect.value)});
    inferButton.addEventListener("click", function() {infer(templateSelect.value, weightNameSelect.textContent)});
}

initEvent();
