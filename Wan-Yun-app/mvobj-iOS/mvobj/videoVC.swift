//
//  videoVC.swift
//  mvobj
//
//  Created by WanYun Sun on 2/6/19.
//  Copyright © 2019 WanYun Sun. All rights reserved.
//

import UIKit
import MobileCoreServices

class videoVC: UIViewController,UINavigationControllerDelegate,UIImagePickerControllerDelegate {

    //button
    @IBOutlet weak var recordBtn: UIButton!
    @IBOutlet weak var chooseBtn: UIButton!
    
    //variables
    var imagePickerController = UIImagePickerController()
    var videoURL: URL?
    let videoFileName = "/video.mp4"
    
    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
    }
    
    @IBAction func recordBtnClicked(_ sender: Any) {
    }
    
    @IBAction func chooseBtnClicked(_ sender: Any) {
        
        
        imagePickerController.sourceType = .savedPhotosAlbum
        imagePickerController.delegate = self
        imagePickerController.mediaTypes = [kUTTypeMovie as String]
        present(imagePickerController, animated: true, completion: nil)
        
    }
    func imagePickerController(picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : AnyObject]) {
        let videoURL = info["UIImagePickerControllerMediaURL"] as? NSURL
        print(videoURL!)
        imagePickerController.dismiss(animated: true, completion: nil)
    }
    
    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}

