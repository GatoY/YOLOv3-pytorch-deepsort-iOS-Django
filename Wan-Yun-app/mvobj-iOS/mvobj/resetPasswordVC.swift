//
//  resetPasswordVC.swift
//  mvobj
//
//  Created by WanYun Sun on 1/5/19.
//  Copyright © 2019 WanYun Sun. All rights reserved.
//

import UIKit
import Alamofire
import SwiftyJSON

class resetPasswordVC: UIViewController {
    
     //textFields
    @IBOutlet weak var emailTxt: UITextField!
    //Buttons
    @IBOutlet weak var ResetBtn: UIButton!
    @IBOutlet weak var CancelBtn: UIButton!
    //URL
    var resetURL = "http://127.0.0.1:8000/api/v1/rest-auth/password/reset/"
    
    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
        
        //background
        let bg = UIImageView(frame: CGRect(x: 0,y: 0,width: self.view.frame.size.width, height: self.view.frame.size.height))
        bg.image = UIImage(named: "bg2.jpg")
        bg.layer.zPosition = -1
        self.view.addSubview(bg)
    }
    
    @IBAction func ResetBtnClicked(_ sender: UIButton) {
        print("Reset button clicked")
        //dismiss keyboard
        self.view.endEditing(true)
        //check if email is given
        if emailTxt.text!.isEmpty{
            //alert msg
            let alert = UIAlertController(title: "Empty field(s)", message: "Please fill email fields.", preferredStyle: UIAlertController.Style.alert)
            let ok = UIAlertAction(title: "OK", style: UIAlertAction.Style.cancel, handler: nil)
            alert.addAction(ok)
            self.present(alert, animated: true, completion: nil)
        }
        //reset password
        //create json
        let email = emailTxt.text!
        let params: [String: Any] = [
            "email":email,
        ]
        //send signup info to server
        AF.request( resetURL,
                    method:.post,
                    parameters: params,
                    encoding: JSONEncoding.default)
            
            .validate()
            .responseJSON { response in
                debugPrint(response)
                
                switch response.result {
                case .success (let data):
                    print("result.success")
                    //alert msg
                    let alert = UIAlertController(title: "Reset Password Email Sent", message: nil, preferredStyle: UIAlertController.Style.alert)
                    let ok = UIAlertAction(title: "OK", style: UIAlertAction.Style.cancel, handler: nil)
                    alert.addAction(ok)
                    self.present(alert, animated: true, completion: nil)
                    return
                    
                case .failure(let error):
                    print("Request failed with error: \(error)")
                    var errormsg = ""
                    if let data = response.data {
                        errormsg = String(data: data, encoding: String.Encoding.utf8) ?? ""
                    }
                    //alert msg
                    let alert = UIAlertController(title: "Signup Fail", message: errormsg as String, preferredStyle: UIAlertController.Style.alert)
                    let ok = UIAlertAction(title: "OK", style: UIAlertAction.Style.cancel, handler: nil)
                    alert.addAction(ok)
                    self.present(alert, animated: true, completion: nil)
                    return
                }
        }
    }
    @IBAction func CancelBtnClicked(_ sender: Any) {
        print("cancel button clicked")
        self.dismiss(animated: true, completion: nil)
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
