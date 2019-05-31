//
//  signupVC.swift
//  mvobj
//
//  Created by WanYun Sun on 1/5/19.
//  Copyright © 2019 WanYun Sun. All rights reserved.
//

import UIKit
import Alamofire
import SwiftyJSON

class signupVC: UIViewController {

    
    //textFields
    @IBOutlet weak var emailTxt: UITextField!
    @IBOutlet weak var passwordTxt: UITextField!
    @IBOutlet weak var rpPasswordTxt: UITextField!
    
    
    //Buttons
    @IBOutlet weak var SignupBtn: UIButton!
    @IBOutlet weak var CancelBtn: UIButton!
    
    //variable
    var token:String = ""
    var signupURL = "http://127.0.0.1:8000/api/v1/rest-auth/registration/"
    
    override func viewDidLoad() {
        super.viewDidLoad()

        
        // Do any additional setup after loading the view.
        self.hideKeyboardWhenTappedAround()
    }
    

    @IBAction func signupBtnClick(_ sender: Any) {
        print("signup button clicked")
        //dismiss keyboard
        self.view.endEditing(true)
        
        //check if all fields are filled
        if emailTxt.text!.isEmpty || passwordTxt.text!.isEmpty || rpPasswordTxt.text!.isEmpty {
            
            //alert msg
            let alert = UIAlertController(title: "Empty field(s)", message: "Please fill all fields.", preferredStyle: UIAlertController.Style.alert)
            let ok = UIAlertAction(title: "OK", style: UIAlertAction.Style.cancel, handler: nil)
            alert.addAction(ok)
            self.present(alert, animated: true, completion: nil)
        }
        //check if repeat password correct
        if passwordTxt.text != rpPasswordTxt.text {
            
            //alert msg
            let alert = UIAlertController(title: "Passwords not match", message: "Repeat Password has to be the same as Password.", preferredStyle: UIAlertController.Style.alert)
            let ok = UIAlertAction(title: "OK", style: UIAlertAction.Style.cancel, handler: nil)
            alert.addAction(ok)
            self.present(alert, animated: true, completion: nil)
        }
        
        //create json
        let email = emailTxt.text!
        let password = passwordTxt.text!
        let params: [String: Any] = [
            "username": email,
            "email":email,
            "password1": password,
            "password2": password
        ]
        //send signup info to server
        AF.request( signupURL,
                   method:.post,
                   parameters: params,
                   encoding: JSONEncoding.default)
            
            .validate()
            .responseJSON { response in
                debugPrint(response)
                
                switch response.result {
                case .success (let data):
                    print("result.success")
                    //store token
                    let json = JSON(data)
                    self.token = json["key"].stringValue
                    print("token: ",self.token)
                    
                    //call login func from AppDelegate, redirect to main page
                    let appDelegate: AppDelegate =  UIApplication.shared.delegate as! AppDelegate
                    appDelegate.login()
                    
                    
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
    
    @IBAction func cancelBtnClick(_ sender: Any) {
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

