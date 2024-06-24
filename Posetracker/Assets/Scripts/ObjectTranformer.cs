using System;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;
using System.IO.Pipes;
using System.Linq;



public class ObjectTransformer : MonoBehaviour
{
    public GameObject skeleton;
    public GameObject heart;
    public GameObject kidney;
    public GameObject handPoint;
 

    public bool anchoredBody = false;

    public int mpIndex = 0;

   public float scale = 2.0f;

    public float x_scale = 8.0f;

    public float y_scale = 6.0f;

    public float z_scale = 1.0f;

    public float x = -4.0f;

    public float y = -9.0f;

    public float z = 7.0f;

    private Vector3 _positionChange;
    private Vector3 handPosition;
    private NamedPipeServerStream server;
    //private NamedPipeServerStream serverOrgan;
    private NamedPipeServerStream serverGesture;
    
    private GameObject heartInstance;
    private GameObject kidneyInstance;

    private string gestureData;
    private bool thumbsUpDetected = false;
    private bool kidneyGesture = false;
    private bool heartGesture = false;
    private bool peaceDetected=false;
  



    void Start()
    {
       

         if (heart != null  && skeleton != null)
         {  
            heartInstance = Instantiate(heart, skeleton.transform);
            kidneyInstance = Instantiate(kidney, skeleton.transform);

            // Set initial positions of heart and kidney relative to skeleton
           
            heartInstance.transform.localPosition = _positionChange + Vector3.up * 1.9f;
            kidneyInstance.transform.localPosition = _positionChange + Vector3.up * 0.8f;
         }


        try
        {
            server = new NamedPipeServerStream("UnityMediaPipeBody", PipeDirection.InOut, 99, PipeTransmissionMode.Message);
            Debug.Log("Waiting for connection on UnityMediaPipeBody...");
            server.WaitForConnection();
            Debug.Log("Connected to UnityMediaPipeBody.");

             // Open the second named pipe.
          
        }
        catch (Exception ex)
        {
            Debug.LogError("Error initializing named pipe server: " + ex.Message);
        }

        try
        {
            serverGesture = new NamedPipeServerStream("UnityMediaPipeGesture", PipeDirection.InOut, 99, PipeTransmissionMode.Message);
            Debug.Log("Waiting for connection on UnityMediaPipeGesture...");
            serverGesture.WaitForConnection();
            Debug.Log("Connected to UnityMediaPipeGesture.");
        }
        
        catch (Exception exy)
        {
             Debug.LogError("Error initializing named pipe server: " + exy.Message);
        }

        // try
        // {
        //     serverOrgan = new NamedPipeServerStream("UnityMediaPipeOrgan", PipeDirection.InOut, 99, PipeTransmissionMode.Message);
        //     Debug.Log("Waiting for connection on UnityMediaPipeOrgan...");
        //     serverOrgan.WaitForConnection();
        //     Debug.Log("Connected to UnityMediaPipeOrgan.");
        // }
        
        // catch (Exception exy)
        // {
        //      Debug.LogError("Error initializing named pipe server: " + exy.Message);
        // }

        
        
    }

 

    void Update()
    {
        try
        {
            var br = new BinaryReader(server, Encoding.UTF8);
            var len = (int)br.ReadUInt32();
            var str = new string(br.ReadChars(len));
            string[] lines = str.Split('\n');

            var gestureReader = new BinaryReader(serverGesture, Encoding.UTF8);
            var len2 = (int)gestureReader.ReadUInt32();
            var str2 = new string(gestureReader.ReadChars(len2));
            string[] gestures = str2.Split('\n');
            foreach (string gesture in gestures)
                {   
                    gestureData = gesture;
                    Debug.Log("Received gesture data: " + gesture);

                    if(gestureData== "thumbs up")
                    {
                        thumbsUpDetected=true;
                    }                  

                }

            
            // var organReader = new BinaryReader(serverOrgan, Encoding.UTF8);
            // var len3 = (int)organReader.ReadUInt32();
            // var str3 = new string(organReader.ReadChars(len3));
            // string[] organs = str3.Split('\n');
           

            
        


            
            foreach (string l in lines)
            {   
                //print(l);
                if (string.IsNullOrWhiteSpace(l))
                    continue;

                string[] s = l.Split('|');

                if (s.Length < 5) continue;


                switch (anchoredBody)
                {
                    case true when s[0] != "ANCHORED":
                    case false when s[0] != "FREE":
                        continue;
                }

                if (!int.TryParse(s[1], out var i)) continue;

               
                if(i==15)
                {
                    handPosition = new Vector3(float.Parse(s[2]), float.Parse(s[3]), float.Parse(s[4]));
                Vector3 screenPos = new Vector3(handPosition.x, 1 - handPosition.y, handPosition.z);
                Vector3 normalizedPos = new Vector3(screenPos.x * x_scale -4.5f , screenPos.y * y_scale -2.5f,
                    screenPos.z * z_scale + z);

                
                   handPosition = new Vector3(normalizedPos.x , normalizedPos.y, normalizedPos.z);
                }
                    
               

                if (i != mpIndex) continue;


                Vector3 position = new Vector3(float.Parse(s[2]), float.Parse(s[3]), float.Parse(s[4]));
             // Convert to screen space coordinates (0-1 range)
                Vector3 screenPosition = new Vector3(position.x, 1 - position.y, position.z);
                Vector3 normalizedPosition = new Vector3(screenPosition.x * x_scale + x, screenPosition.y * y_scale + y,
                    screenPosition.z * z_scale + z);

                
                _positionChange = new Vector3(normalizedPosition.x , normalizedPosition.y, normalizedPosition.z);
                
                if (thumbsUpDetected && Vector3.Distance(handPosition, heartInstance.transform.position) < 0.6f)
                {   
                        heartGesture = false;
                        heartInstance.transform.parent = null;
                        heartInstance.transform.position = handPosition;

                   
                    if (gestureData == "peace")
                    {
                        thumbsUpDetected = false;
                        heartGesture = true;
                        peaceDetected = true;
                        heartInstance.transform.parent = null;
                        
                                
                    }

                    if(gestureData=="thumbs down")
                    {   
                        thumbsUpDetected = false;
                        peaceDetected =false;
                        heartGesture= false;
                        heartInstance.transform.localScale = new Vector3(0.4f, 0.4f, 0.4f);
                        heartInstance.transform.rotation = Quaternion.identity;
                        heartInstance.transform.localPosition = _positionChange + Vector3.up * 1.9f;
                        heartInstance.transform.parent = skeleton.transform;
                        
                    }

                   
                    
                }

                if(peaceDetected && heartGesture)
                         {
                            gesturefunction(heartInstance,gestureData);
                         }

                

                 if (thumbsUpDetected && Vector3.Distance(handPosition, kidneyInstance.transform.position) < 0.6f)
                {   
                    kidneyGesture = false;
                    kidneyInstance.transform.parent = null;
                    kidneyInstance.transform.position = handPosition;

                    if (gestureData == "peace")
                    {
                        thumbsUpDetected = false;
                        kidneyGesture = true;
                        peaceDetected = true;
                        kidneyInstance.transform.parent = null;
                    }

                    if(gestureData=="thumbs down")
                    {   
                        thumbsUpDetected = false;
                        peaceDetected =false;
                        kidneyGesture= false;
                        kidneyInstance.transform.localScale = new Vector3(0.5f, 0.5f, 0.5f);
                        kidneyInstance.transform.rotation = Quaternion.identity;
                        kidneyInstance.transform.localPosition = _positionChange + Vector3.up * 0.8f;
                        kidneyInstance.transform.parent = skeleton.transform;
                        
                    }

                    

                }

                 if(peaceDetected && kidneyGesture)
                         {
                            gesturefunction(kidneyInstance,gestureData);
                         }
                
           }

        }
         
        catch (Exception ex)
        {
            Debug.LogError("Error reading from named pipe: " + ex.Message);
        }

         skeleton.transform.position = _positionChange;
         skeleton.transform.localScale = new Vector3(scale, scale, scale);


       

        
    }

    private void gesturefunction(GameObject organInstance, string gestureData)
    {
        if (gestureData == "rock")
        {
            // Rotate the heart when 'thumbs up' gesture is detected
            if (organInstance != null)
            {
                organInstance.transform.Rotate(Vector3.up, Time.deltaTime * 90f); // Rotate 90 degrees per second around the y-axis
            }
        }
         if (gestureData== "live long" || gestureData=="stop")
        {
            // Increase heart size when 'stop' gesture is detected
            if (organInstance != null)
            {
                organInstance.transform.localScale += new Vector3(0.05f, 0.05f, 0.05f);
            }
        }

        if (gestureData == "fist")
        {
            // Decrease heart size when 'fist' gesture is detected
            if (organInstance != null)
            {
                organInstance.transform.localScale -= new Vector3(0.05f, 0.05f, 0.05f);
            }
        }


    }



   
    void OnDestroy()
    {
       if (server != null)
        {
            server.Close();
            server.Dispose();
        }
        if (serverGesture != null)
        {
            serverGesture.Close();
            serverGesture.Dispose();
        }
        // if (serverOrgan != null)
        // {
        //     serverOrgan.Close();
        //     serverOrgan.Dispose();
        // }
    }
}