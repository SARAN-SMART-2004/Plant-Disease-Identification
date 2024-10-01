let text = document.getElementById('text');
let leaf = document.getElementById('leaf');
let hill1 = document.getElementById('hill1');
let hill4 = document.getElementById('hill4');
let hill5 = document.getElementById('hill5');
let camera = document.getElementById('camera');

window.addEventListener('scroll', () => {
    let value = window.scrollY;
    text.style.marginTop = value * 2.5 + 'px';
    camera.style.marginTop = value * 2.5 + 'px';
    leaf.style.top = value * -1.5 + 'px';
    leaf.style.left = value * 1.5 + 'px';
    hill5.style.left = value * 1.5 + 'px';
    hill4.style.left = value * -1.5 + 'px';
    hill1.style.top = value * 1 + 'px';
});


let zone = document.getElementById('butt');
let cameraa = document.getElementById("enhancerUIContainer");
let frame = document.getElementById('framepic');
let i = 0;

// function display(){
//   const form = document.getElementById('uploadForm');
//   const formData = new FormData(form);

//   fetch('/display', {
    
//     body: file
//   })
//   .then(response => response.json())
//   .then(data => {
//     // Handle the response data if needed
//     console.log(data);

//     // Redirect to the '/display' page
//     window.location.href = '/display';
//   })
//   .catch(error => {
//     // Handle errors
//     console.error('Error:', error);
//   });
// }




function cameras() {
  if(i==0){
  i=1;
  if(i!=0){
    cameraa.style.display = "block";
    cameraa.style.border = "3px solid yellow"
    framepic.style.display = "none";
    zone.style.display = "none";
    
    let enhancer = null;
    (async () => {
        enhancer = await Dynamsoft.DCE.CameraEnhancer.createInstance();
        document.getElementById("enhancerUIContainer").appendChild(enhancer.getUIElement());
        await enhancer.open(true);
        document.querySelector(".dce-btn-close").onclick = ()=>{
          framepic.style.display = "block";
          i=0;
        }
    })();
    let save = document.getElementById('capture');
    save.onclick = () => {
        if (enhancer) {
            let frame = enhancer.getFrame();
    
            let width = screen.availWidth;
            let height = screen.availHeight;
            let popW = 800, popH = 500;
            let left = (width - popW) / 2;
            let top = (height - popH) / 2;
    
            popWindow = window.open('', 'popup', 'width=' + popW + ',height=' + popH +
                ',top=' + top + ',left=' + left + ', scrollbars=yes');
                console.log("%c",save.value);
            popWindow.document.body.appendChild(frame.canvas);
            
        }
    };
  } 
}
}

function opens() {
  if(i==0){
    zone.style.display = "block";
    cameraa.style.display = "none";
  }
}

function closes() {
    zone.style.display = "none";
    cameraa.style.display = "block";

}


function reveal() {
    var reveals = document.querySelectorAll(".reveal");
  
    for (var i = 0; i < reveals.length; i++) {
      var windowHeight = window.innerHeight;
      var elementTop = reveals[i].getBoundingClientRect().top;
      var elementVisible = 150;
      
      if (elementTop < windowHeight - elementVisible) {
        reveals[i].classList.add("active");
      } else {
        reveals[i].classList.remove("active");
      }
    }
  }
  
  window.addEventListener("scroll", reveal);

