const PROJECT = 'all'; // try 'weurope', 'canada'…
const API_URL = 'https://my-api.plantnet.org/v2/identify/' + PROJECT;


// to make this example work you have to expose your API key and
// authorize your webserver address in "Authorized domains" section
// see https://my.plantnet.org/account/doc#exposekey
const API_KEY = '2b10AgX9LS0oGoIWZeDphXFbqO';

const identify = async () => {
    // 1. Get the file from an input type=file : 
    const fileInput = document.getElementById('file');
    const images = fileInput.files;
    if (images.length === 0) {
        console.error('choose a file');
        return false;
    }

    // 2. Build POST form data
    const form = new FormData();
    for (let i = 0; i < images.length; i += 1) {
        form.append('organs', 'auto');
        form.append('images', images[i]);
    }

    // 3. Add GET URL parameters
    const url = new URL(API_URL);
    url.searchParams.append('include-related-images', 'true'); // try false
    url.searchParams.append('api-key', API_KEY);

    // 4. Send request
    fetch(url.toString(), {
        method: 'POST',
        body: form,
    })
    .then((response) => {
        if (response.ok) {
            response.json()
            .then((r) => {
               
                document.getElementById('common_name').innerHTML = JSON.stringify(r.bestMatch);
                document.getElementById('images').innerHTML = JSON.stringify((r.results[0].score));
                document.getElementById('family').innerHTML = JSON.stringify(r.results[0].species.commonNames[0]);
                let family = document.getElementById('family');
    console.log(JSON.stringify(r.results[0].species.commonNames[0]))
    
                document.getElementById("img3").src=r.results[0].images[0].url.s;            })
            .catch(console.error);
        } else {
            const resp = `status: ${response.status} (${response.statusText})`;
            document.getElementById('results').innerHTML = resp;
        }
    })
    .catch((error) => {
        console.error(error);
    });
};
document.addEventListener('DOMContentLoaded', () => {

    const form = document.getElementById('myform');
    form.addEventListener('submit', (evt) => {
        evt.preventDefault();
        identify();
    });

});


 let neem = document.getElementById("neem");
 let avacado = document.getElementById("avacado");
 let aswakandha = document.getElementById("aswakandha");
 let pepper = document.getElementById("pepper");
 let map = document.getElementById("map");
 if(fam==neems){
    neem.style.display="block";
    avacado.style.display="none";
    aswakandha.style.display="none";
    pepper.style.display="none";
    map.style.display="none";
 }
 else if(fam==""){
    neem.style.display="none";
    avacado.style.display="none";
    aswakandha.style.display="none";
    pepper.style.display="none";
    map.style.display="none";
 }
 else if(fam=="avacado"|| fam=="Avacado"){
    neem.style.display="none";
    avacado.style.display="block";
    aswakandha.style.display="none";
    pepper.style.display="none";
    map.style.display="none";
 }
 else if(fam=="aswakandha"|| fam=="Aswakandha"){
    neem.style.display="none";
    avacado.style.display="none";
    aswakandha.style.display="block";
    pepper.style.display="none";
    map.style.display="none";
 }
 else if(fam=="pepper"|| fam=="Pepper"){
    neem.style.display="none";
    avacado.style.display="none";
    aswakandha.style.display="none";
    pepper.style.display="block";
    map.style.display="none";
 }
 else{
    neem.style.display="none";
    avacado.style.display="none";
    aswakandha.style.display="none";
    pepper.style.display="none";
    map.style.display="block";
 }
 let ms = document.getElementById("ms");
 ms.onclick=()=>{
    neem.style.display="block";
 }

 