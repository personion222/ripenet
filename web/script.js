// var corsAttr = new EnableCorsAttribute("*", "*", "*");
// config.EnableCors(corsAttr);

var model;
const body = document.getElementsByTagName("body");
const top_cont = document.getElementById("top");
var ripesfx = new Audio("ripe.wav");
var unripesfx = new Audio("unripe.wav");
const video = document.getElementById("cam");
const canvas = document.getElementById("nn-feed");
const ctx = canvas.getContext("2d", {willReadFrequently: true});
const thresh_label = document.getElementById("thresh");
const spinner = document.createElement("span");
var ripe_button = document.getElementById("ripe");
var calculating = false;
const target_width = 64;
const target_height = 64;
var aspect;
var img_arr;
var raw_arr;
var tensor_img;
var ripe_val;
var startx;
var starty;
var scalex;
var scaley;
var cur_thresh;
var wrapper;

spinner.classList.add("spinner-border");
spinner.classList.add("spinner-border-sm");

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function init_model () {
    model = await tf.loadGraphModel("models/ripe_single/model.json");
}

init_model();

function sigmoid(x, c, s) {
    return 1 / (1 + Math.pow(2.718, c - 2 * c * (x - s)));
}

function append_alert(message, type, time) {
    wrapper = document.createElement('div');
    wrapper.innerHTML = [
      `<div class="alert alert-${type} alert-dismissible" role="alert">`,
      `   <div>${message}</div>`,
      '   <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>',
      '</div>'
    ].join('');
    top_cont.append(wrapper);
    setTimeout(() => {
        wrapper.remove();
    }, time)
}

function create_3d_arr(x, y, z) {
    var arr = new Array(x);
    for (let i = 0; i < x; i += 1) {
        arr[i] = new Array(y);
        for (let j = 0; j < y; j += 1) {
            arr[i][j] = new Uint8Array(3);
        }
    }

    return arr;
}

function start_cam() {
    const constraints = {
        video: {
            facingMode: "environment"
        }
    };

    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        video.srcObject = stream
    });
}

function inc_thresh(inc_val) {
    thresh_label.innerHTML = (parseFloat(thresh_label.innerHTML) + inc_val).toFixed(1);
    if (parseFloat(thresh_label.innerHTML) < 0) {
        thresh_label.innerHTML = "0.0";
    } else if (parseFloat(thresh_label.innerHTML) > 1) {
        thresh_label.innerHTML = "1.0";
    }
}

async function get_ripeness() {
    calculating = true;
    ripe_button.disabled = true;
    ripe_button.appendChild(spinner);
    aspect = video.videoWidth / video.videoHeight;

    if (aspect > 1) {
        starty = 0;
        scaley = target_height;
        startx = -((aspect - 1) / 2 * target_width);
        scalex = target_width * aspect;
    } else {
        startx = 0;
        scalex = target_width;
        starty = -((1 / aspect - 1) / 2 * target_height);
        scaley = target_height * aspect;
    }

    ctx.drawImage(video, startx, starty, scalex, scaley);
    raw_arr = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

    img_arr = create_3d_arr(target_width, target_height, 3);

    for (let i = 0; i < raw_arr.length; i += 4) {
        img_arr[Math.floor(i / (4 * target_width)) % target_height][Math.floor(i / 4) % target_height][0] = raw_arr[i];
        img_arr[Math.floor(i / (4 * target_width)) % target_height][Math.floor(i / 4) % target_height][1] = raw_arr[i + 1];
        img_arr[Math.floor(i / (4 * target_width)) % target_height][Math.floor(i / 4) % target_height][2] = raw_arr[i + 2];
    }

    tensor_img = tf.div(tf.tensor(img_arr), 255);
    ripe_val = sigmoid(model.predict(tensor_img.reshape([1, 64, 64, 3])).dataSync()[0], 4, 0.25);
    console.log(ripe_val);

    cur_thresh = parseFloat(thresh_label.innerHTML);

    await sleep(500);

    if (ripe_val > cur_thresh) {
        ripesfx.play();
        append_alert("ripe :) " + ripe_val.toFixed(2) * 100 + "%", "warning", 2000);
    } else {
        unripesfx.play();
        append_alert("unripe :( " + (1 - ripe_val.toFixed(1)) * 100 + "%", "success", 2000);
    }

    calculating = false;
    ripe_button.disabled = false;
    ripe_button.removeChild(spinner);
}

start_cam();
